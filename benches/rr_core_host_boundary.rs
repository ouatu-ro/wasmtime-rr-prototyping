#![cfg(feature = "rr")]

use core::any::Any;
use criterion::{Criterion, criterion_group, criterion_main};
use std::io::{Cursor, sink};
use std::time::Instant;
use wasmtime::{
    Config, Engine, Linker, Module, OptLevel, RRConfig, RecordSettings, ReplayEnvironment,
    ReplaySettings, Store,
};

criterion_group!(benches, rr_core_host_boundary);
criterion_main!(benches);

const MICRO_CALLS: u64 = 1_000;
const MACRO_CALLS: u64 = 100_000;

fn rr_core_host_boundary(c: &mut Criterion) {
    bench_profile(c, "micro", MICRO_CALLS);
    bench_profile(c, "macro", MACRO_CALLS);
}

fn bench_profile(c: &mut Criterion, label: &str, calls: u64) {
    let module_wat = r#"
        (module
            (import "env" "nop" (func $nop))
            (func (export "run") (param i64)
                loop
                    call $nop
                    local.get 0
                    i64.const -1
                    i64.add
                    local.tee 0
                    i64.const 0
                    i64.ne
                    br_if 0
                end
            )
        )
    "#;

    let disabled_engine = create_engine(RRConfig::None);
    let disabled_module = Module::new(&disabled_engine, module_wat).unwrap();
    let mut disabled_linker = Linker::new(&disabled_engine);
    disabled_linker.func_wrap("env", "nop", || {}).unwrap();
    let mut disabled_store = Store::new(&disabled_engine, ());
    let disabled_instance = disabled_linker
        .instantiate(&mut disabled_store, &disabled_module)
        .unwrap();
    let disabled_run = disabled_instance
        .get_typed_func::<u64, ()>(&mut disabled_store, "run")
        .unwrap();

    c.bench_function(&format!("rr-core-host-boundary/{label}/disabled"), |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                disabled_run.call(&mut disabled_store, calls).unwrap();
            }
            start.elapsed()
        })
    });

    let recording_engine = create_engine(RRConfig::Recording);
    let recording_module = Module::new(&recording_engine, module_wat).unwrap();
    let mut recording_linker = Linker::new(&recording_engine);
    recording_linker.func_wrap("env", "nop", || {}).unwrap();
    let mut recording_store = Store::new(&recording_engine, ());
    recording_store
        .record(sink(), RecordSettings::default())
        .unwrap();
    let recording_instance = recording_linker
        .instantiate(&mut recording_store, &recording_module)
        .unwrap();
    let recording_run = recording_instance
        .get_typed_func::<u64, ()>(&mut recording_store, "run")
        .unwrap();

    c.bench_function(&format!("rr-core-host-boundary/{label}/recording"), |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                recording_run.call(&mut recording_store, calls).unwrap();
            }
            start.elapsed()
        })
    });

    let trace = record_trace(module_wat, calls);
    let replay_engine = create_engine(RRConfig::Replaying);
    let replay_module = Module::new(&replay_engine, module_wat).unwrap();

    c.bench_function(&format!("rr-core-host-boundary/{label}/replay"), |b| {
        b.iter_custom(|iters| {
            let mut elapsed = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut renv = ReplayEnvironment::new(&replay_engine, ReplaySettings::default());
                renv.add_module(replay_module.clone());
                let mut replay = renv
                    .instantiate_with(
                        Cursor::new(trace.clone()),
                        |_| Ok(()),
                        |linker| {
                            linker.allow_shadowing(true);
                            linker.func_wrap("env", "nop", || -> wasmtime::Result<()> {
                                panic!("host function body should not execute during replay")
                            })?;
                            Ok(())
                        },
                        |_| Ok(()),
                    )
                    .unwrap();
                let start = Instant::now();
                replay.run_to_completion().unwrap();
                elapsed += start.elapsed();
            }
            elapsed
        })
    });
}

fn create_engine(rr: RRConfig) -> Engine {
    let mut config = Config::new();
    config
        .debug_info(false)
        .cranelift_opt_level(OptLevel::None)
        .rr(rr);
    Engine::new(&config).unwrap()
}

fn record_trace(module_wat: &str, calls: u64) -> Vec<u8> {
    let engine = create_engine(RRConfig::Recording);
    let module = Module::new(&engine, module_wat).unwrap();
    let mut linker = Linker::new(&engine);
    linker.func_wrap("env", "nop", || {}).unwrap();

    let mut store = Store::new(&engine, ());
    store
        .record(Cursor::new(Vec::new()), RecordSettings::default())
        .unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();
    let run = instance
        .get_typed_func::<u64, ()>(&mut store, "run")
        .unwrap();
    run.call(&mut store, calls).unwrap();

    let writer = store.into_record_writer().unwrap();
    let trace = (writer as Box<dyn Any>)
        .downcast::<Cursor<Vec<u8>>>()
        .expect("trace writer should be an in-memory cursor");
    trace.into_inner()
}
