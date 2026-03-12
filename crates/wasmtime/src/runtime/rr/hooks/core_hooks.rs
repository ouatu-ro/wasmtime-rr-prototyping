#[cfg(feature = "rr")]
use super::{ActiveBoundaryMode, active_boundary_mode, replay_data_from_store, replay_data_from_store_mut};
use crate::rr::FlatBytes;
#[cfg(feature = "rr")]
use crate::rr::{
    RREvent, RRFuncArgVals, RRFuncArgValsConvertable, ReplayError, Replayer, ResultEvent, Validate,
    common_events::{HostFuncEntryEvent, HostFuncReturnEvent, WasmFuncReturnEvent},
    core_events::{InstantiationEvent, WasmFuncEntryEvent},
};
use crate::store::{InstanceId, StoreOpaque};
use crate::{Caller, FuncType, Module, StoreContextMut, ValRaw, WasmFuncOrigin, prelude::*};
#[cfg(feature = "rr")]
use wasmtime_environ::EntityIndex;
use wasmtime_environ::WasmChecksum;

/// Store-local boundary protocol for core wasm <-> host transitions.
#[cfg(feature = "rr")]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct CoreHostBoundaryProtocol {
    mode: ActiveBoundaryMode,
}

#[cfg(feature = "rr")]
impl CoreHostBoundaryProtocol {
    #[inline(always)]
    fn maybe_for_store(store: &StoreOpaque) -> Option<Self> {
        Some(Self {
            mode: active_boundary_mode(store)?,
        })
    }

    #[inline(always)]
    fn on_host_entry<T>(
        self,
        args: &[T],
        flat: impl Iterator<Item = u8>,
        store: &mut StoreOpaque,
    ) -> Result<()>
    where
        T: FlatBytes,
    {
        match self.mode {
            ActiveBoundaryMode::Recording => Self::record_host_entry(args, flat, store),
            ActiveBoundaryMode::Replaying => Self::replay_host_entry(args, flat, store),
        }
    }

    #[inline(always)]
    fn on_host_return_record<T>(
        self,
        args: &[T],
        flat: impl Iterator<Item = u8>,
        store: &mut StoreOpaque,
    ) -> Result<()>
    where
        T: FlatBytes,
    {
        if self.mode == ActiveBoundaryMode::Recording {
            Self::record_host_return(args, flat, store)?;
        }
        Ok(())
    }

    #[inline(always)]
    fn on_host_return_replay_substitute<T, U: 'static>(
        self,
        args: &mut [T],
        caller: &mut Caller<'_, U>,
    ) -> Result<()>
    where
        T: FlatBytes,
    {
        match self.mode {
            ActiveBoundaryMode::Replaying => Self::substitute_replay_host_return(args, caller),
            ActiveBoundaryMode::Recording => Ok(()),
        }
    }

    #[cold]
    fn record_host_entry<T>(
        args: &[T],
        flat: impl Iterator<Item = u8>,
        store: &mut StoreOpaque,
    ) -> Result<()>
    where
        T: FlatBytes,
    {
        store.record_event_validation(|| HostFuncEntryEvent {
            args: RRFuncArgVals::from_flat_iter(args, flat),
        })?;
        Ok(())
    }

    #[cold]
    fn replay_host_entry<T>(
        args: &[T],
        flat: impl Iterator<Item = u8>,
        store: &mut StoreOpaque,
    ) -> Result<()>
    where
        T: FlatBytes,
    {
        store.next_replay_event_validation::<HostFuncEntryEvent, _, _>(|| HostFuncEntryEvent {
            args: RRFuncArgVals::from_flat_iter(args, flat),
        })?;
        Ok(())
    }

    #[cold]
    fn record_host_return<T>(
        args: &[T],
        flat: impl Iterator<Item = u8>,
        store: &mut StoreOpaque,
    ) -> Result<()>
    where
        T: FlatBytes,
    {
        store.record_event(|| HostFuncReturnEvent {
            args: RRFuncArgVals::from_flat_iter(args, flat),
        })?;
        Ok(())
    }

    #[cold]
    fn substitute_replay_host_return<T, U: 'static>(
        args: &mut [T],
        caller: &mut Caller<'_, U>,
    ) -> Result<()>
    where
        T: FlatBytes,
    {
        loop {
            let event = caller
                .store
                .0
                .replay_buffer_mut()
                .expect("replay mode requires a replay buffer")
                .next_event()?;

            match event {
                RREvent::HostFuncReturn(event) => {
                    event.args.into_raw_slice(args);
                    return Ok(());
                }
                RREvent::CoreWasmFuncEntry(event) => {
                    Self::replay_reentrant_wasm_call(event, caller)?;
                }
                other => {
                    bail!(
                        "Unexpected event during core wasm host function replay: expected HostFuncReturn or CoreWasmFuncEntry, got {other:?}",
                    );
                }
            }
        }
    }

    #[cold]
    fn replay_reentrant_wasm_call<U: 'static>(
        event: WasmFuncEntryEvent,
        caller: &mut Caller<'_, U>,
    ) -> Result<()> {
        let entity = EntityIndex::from(event.func_index);

        // The replay mode on this protocol guarantees the replay context.
        let replay_data = unsafe { replay_data_from_store(&caller.store) };

        let instance = replay_data.get_module_instance(event.instance)?;
        let mut store = &mut caller.store;
        let func = instance
            ._get_export(store.0, entity)
            .into_func()
            .ok_or(ReplayError::InvalidCoreFuncIndex(entity))?;

        let params_ty = func.ty(&store).params().collect::<Vec<_>>();
        let mut results = vec![crate::Val::I64(0); func.ty(&store).results().len()];
        let params = event.args.to_val_vec(&mut store, params_ty);

        func.call_impl_check_args(&mut store, &params, &mut results)?;
        unsafe {
            func.call_impl_do_call(&mut store, params.as_slice(), results.as_mut_slice())?;
        }
        Ok(())
    }
}

/// Record and replay hook operation for core wasm function entry events
///
/// Recording/replay validation DOES NOT happen if origin is `None`
#[inline(always)]
pub fn record_and_replay_validate_wasm_func<F, T>(
    wasm_call: F,
    args: &[ValRaw],
    ty: &FuncType,
    origin: Option<WasmFuncOrigin>,
    store: &mut StoreContextMut<'_, T>,
) -> Result<()>
where
    F: FnOnce(&mut StoreContextMut<'_, T>) -> Result<()>,
{
    let _ = (args, ty, origin);
    #[cfg(feature = "rr")]
    {
        if let Some(origin) = origin {
            store.0.record_event(|| {
                let flat = ty.params().map(|t| t.to_wasm_type().byte_size());
                WasmFuncEntryEvent {
                    instance: origin.instance.into(),
                    func_index: origin.index.into(),
                    args: RRFuncArgVals::from_flat_iter(args, flat),
                }
            })?;
        }
    }
    let result = wasm_call(store);
    #[cfg(feature = "rr")]
    {
        if origin.is_some() {
            if let Err(e) = &result {
                log::warn!("Wasm function call exited with error: {e:?}");
            }
            let flat = ty.results().map(|t| t.to_wasm_type().byte_size());
            let result = result.map(|_| RRFuncArgVals::from_flat_iter(args, flat));
            store.0.record_event_validation(|| {
                WasmFuncReturnEvent(ResultEvent::from_anyhow_result(&result))
            })?;
            store
                .0
                .next_replay_event_validation::<WasmFuncReturnEvent, _, &Result<RRFuncArgVals>>(
                    || &result,
                )?;
            result?;
            Ok(())
        } else {
            result
        }
    }
    #[cfg(not(feature = "rr"))]
    {
        result
    }
}

/// Record hook operation for host function entry events
#[inline(always)]
pub fn record_validate_host_func_entry<T>(
    args: &[T],
    flat: impl Iterator<Item = u8>,
    store: &mut StoreOpaque,
) -> Result<()>
where
    T: FlatBytes,
{
    let _ = (args, &flat, &store);
    #[cfg(feature = "rr")]
    {
        if let Some(protocol) = CoreHostBoundaryProtocol::maybe_for_store(store) {
            protocol.on_host_entry(args, flat, store)?;
        }
    }
    Ok(())
}

/// Record hook operation for host function return events
#[inline(always)]
pub fn record_host_func_return<T>(
    args: &[T],
    flat: impl Iterator<Item = u8>,
    store: &mut StoreOpaque,
) -> Result<()>
where
    T: FlatBytes,
{
    let _ = (args, &flat, &store);
    #[cfg(feature = "rr")]
    {
        if let Some(protocol) = CoreHostBoundaryProtocol::maybe_for_store(store) {
            protocol.on_host_return_record(args, flat, store)?;
        }
    }
    Ok(())
}

/// Replay hook operation for host function entry events
#[inline(always)]
pub fn replay_validate_host_func_entry<T>(
    args: &[T],
    flat: impl Iterator<Item = u8>,
    store: &mut StoreOpaque,
) -> Result<()>
where
    T: FlatBytes,
{
    let _ = (args, &flat, &store);
    #[cfg(feature = "rr")]
    {
        if let Some(protocol) = CoreHostBoundaryProtocol::maybe_for_store(store) {
            protocol.on_host_entry(args, flat, store)?;
        }
    }
    Ok(())
}

/// Replay hook operation for host function return events.
#[inline(always)]
pub fn replay_host_func_return<T, U: 'static>(
    args: &mut [T],
    caller: &mut Caller<'_, U>,
) -> Result<()>
where
    T: FlatBytes,
{
    #[cfg(feature = "rr")]
    {
        if let Some(protocol) = CoreHostBoundaryProtocol::maybe_for_store(caller.store.0) {
            protocol.on_host_return_replay_substitute(args, caller)?;
        }
    }
    let _ = (args, caller);
    Ok(())
}

/// Hook for recording a module instantiation event and validating the
/// instantiation during replay.
pub fn record_and_replay_validate_instantiation<T: 'static>(
    store: &mut StoreContextMut<'_, T>,
    module: WasmChecksum,
    instance: InstanceId,
) -> Result<()> {
    #[cfg(feature = "rr")]
    {
        store.0.record_event(|| InstantiationEvent {
            module,
            instance: instance.into(),
        })?;
        if store.0.replay_enabled() {
            let replay_data = unsafe { replay_data_from_store_mut(store) };
            replay_data.take_current_module_instantiation().expect(
                "replay driver should have set module instantiate data before trying to validate it",
            ).validate(&InstantiationEvent { module, instance: instance.into() })?;
        }
    }
    let _ = (store, module, instance);
    Ok(())
}

/// Ensure that memories are not exported memories in Core wasm modules when
/// recording is enabled.
pub fn rr_validate_module_unexported_memory(module: &Module) -> Result<()> {
    // Check for exported memories when recording is enabled.
    #[cfg(feature = "rr")]
    {
        if module.engine().is_recording()
            && module.exports().any(|export| {
                if let crate::ExternType::Memory(_) = export.ty() {
                    true
                } else {
                    false
                }
            })
        {
            bail!("Cannot support recording for core wasm modules when a memory is exported");
        }
    }
    let _ = module;
    Ok(())
}

#[cfg(all(test, feature = "rr"))]
mod tests {
    use super::*;
    use crate::rr::{RREvent, ReplayBuffer, Replayer, common_events};
    use crate::{
        Config, Engine, Linker, RRConfig, RecordSettings, ReplayEnvironment, ReplaySettings, Store,
    };
    use core::any::Any;
    use std::io::Cursor;
    use std::io::sink;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::{Duration, Instant};

    fn create_engine(rr: RRConfig) -> Result<Engine> {
        let mut config = Config::new();
        config
            .debug_info(true)
            .cranelift_opt_level(crate::OptLevel::None);
        config.rr(rr);
        Engine::new(&config)
    }

    fn take_trace_cursor<T>(store: Store<T>) -> Result<Cursor<Vec<u8>>> {
        let trace_box = store.into_record_writer()?;
        let any_box: Box<dyn Any> = trace_box;
        let mut trace_reader = any_box.downcast::<Cursor<Vec<u8>>>().unwrap();
        trace_reader.set_position(0);
        Ok(*trace_reader)
    }

    fn collect_replay_events(trace: Cursor<Vec<u8>>) -> Result<Vec<RREvent>> {
        let replay = <ReplayBuffer as Replayer>::new_replayer(trace, ReplaySettings::default())?;
        replay
            .collect::<Result<Vec<_>, ReplayError>>()
            .map_err(Into::into)
    }

    #[test]
    fn rr_disabled_host_boundary_parity() -> Result<()> {
        let engine = create_engine(RRConfig::None)?;
        let module = Module::new(
            &engine,
            r#"
                (module
                    (import "env" "double" (func $double (param i32) (result i32)))
                    (func (export "main") (param i32) (result i32)
                        local.get 0
                        call $double
                        call $double
                    )
                )
            "#,
        )?;

        let mut linker = Linker::new(&engine);
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        linker.func_wrap("env", "double", move |param: i32| {
            seen.fetch_add(1, Ordering::Relaxed);
            param * 2
        })?;

        let mut store = Store::new(&engine, ());
        let instance = linker.instantiate(&mut store, &module)?;
        let main = instance.get_typed_func::<i32, i32>(&mut store, "main")?;

        assert_eq!(main.call(&mut store, 21)?, 84);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        Ok(())
    }

    #[test]
    fn recording_sequence_parity() -> Result<()> {
        let engine = create_engine(RRConfig::Recording)?;
        let module_wat = r#"
            (module
                (import "env" "double" (func $double (param i32) (result i32)))
                (func (export "main") (param i32) (result i32)
                    local.get 0
                    call $double
                )
            )
        "#;
        let module = Module::new(&engine, module_wat)?;

        let mut linker = Linker::new(&engine);
        linker.func_wrap("env", "double", |param: i32| param * 2)?;

        let mut store = Store::new(&engine, ());
        store.record(
            Cursor::new(Vec::new()),
            RecordSettings {
                add_validation: true,
                ..Default::default()
            },
        )?;

        let instance = linker.instantiate(&mut store, &module)?;
        let main = instance.get_typed_func::<i32, i32>(&mut store, "main")?;
        assert_eq!(main.call(&mut store, 21)?, 42);

        let events = collect_replay_events(take_trace_cursor(store)?)?;
        let classes = events
            .iter()
            .map(|event| match event {
                RREvent::CoreWasmInstantiation(_) => "CoreWasmInstantiation",
                RREvent::CoreWasmFuncEntry(_) => "CoreWasmFuncEntry",
                RREvent::HostFuncEntry(_) => "HostFuncEntry",
                RREvent::HostFuncReturn(_) => "HostFuncReturn",
                RREvent::WasmFuncReturn(_) => "WasmFuncReturn",
                other => panic!("unexpected event in trace: {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            classes,
            vec![
                "CoreWasmInstantiation",
                "CoreWasmFuncEntry",
                "HostFuncEntry",
                "HostFuncReturn",
                "WasmFuncReturn",
            ]
        );
        Ok(())
    }

    #[test]
    fn replay_return_substitution_parity() -> Result<()> {
        let module_wat = r#"
            (module
                (import "env" "double" (func $double (param i32) (result i32)))
                (func (export "main") (param i32) (result i32)
                    local.get 0
                    call $double
                )
            )
        "#;

        let engine = create_engine(RRConfig::Recording)?;
        let module = Module::new(&engine, module_wat)?;
        let mut linker = Linker::new(&engine);
        linker.func_wrap("env", "double", |param: i32| param * 2)?;

        let mut store = Store::new(&engine, ());
        store.record(
            Cursor::new(Vec::new()),
            RecordSettings {
                add_validation: true,
                ..Default::default()
            },
        )?;
        let instance = linker.instantiate(&mut store, &module)?;
        let main = instance.get_typed_func::<i32, i32>(&mut store, "main")?;
        assert_eq!(main.call(&mut store, 21)?, 42);
        let trace = take_trace_cursor(store)?;

        let replay_engine = create_engine(RRConfig::Replaying)?;
        let replay_module = Module::new(&replay_engine, module_wat)?;
        let mut renv = ReplayEnvironment::new(&replay_engine, ReplaySettings::default());
        renv.add_module(replay_module);

        let mut replay_instance = renv.instantiate_with(
            trace,
            |_| Ok(()),
            |linker| {
                linker.allow_shadowing(true);
                linker.func_wrap("env", "double", |_param: i32| -> crate::Result<i32> {
                    panic!("host function body should not execute during replay")
                })?;
                Ok(())
            },
            |_| Ok(()),
        )?;
        replay_instance.run_to_completion()?;
        Ok(())
    }

    #[test]
    fn reentrant_replay_trace_remains_balanced() -> Result<()> {
        let module_wat = r#"
            (module
                (import "env" "host_call" (func $host_call (param i32) (result i32)))
                (func (export "main") (param i32) (result i32)
                    local.get 0
                    call $host_call
                )
                (func (export "wasm_callback") (param i32) (result i32)
                    local.get 0
                    i32.const 1
                    i32.add
                )
            )
        "#;

        let engine = create_engine(RRConfig::Recording)?;
        let module = Module::new(&engine, module_wat)?;
        let mut linker = Linker::new(&engine);
        linker.func_wrap(
            "env",
            "host_call",
            |mut caller: crate::Caller<'_, ()>, param: i32| -> crate::Result<i32> {
                let func = caller
                    .get_export("wasm_callback")
                    .unwrap()
                    .into_func()
                    .unwrap();
                let typed = func.typed::<i32, i32>(&caller)?;
                typed.call(&mut caller, param)
            },
        )?;

        let mut store = Store::new(&engine, ());
        store.record(
            Cursor::new(Vec::new()),
            RecordSettings {
                add_validation: true,
                ..Default::default()
            },
        )?;
        let instance = linker.instantiate(&mut store, &module)?;
        let main = instance.get_typed_func::<i32, i32>(&mut store, "main")?;
        assert_eq!(main.call(&mut store, 42)?, 43);

        let events = collect_replay_events(take_trace_cursor(store)?)?;
        let classes = events
            .iter()
            .map(|event| match event {
                RREvent::CoreWasmInstantiation(_) => "CoreWasmInstantiation",
                RREvent::CoreWasmFuncEntry(_) => "CoreWasmFuncEntry",
                RREvent::HostFuncEntry(_) => "HostFuncEntry",
                RREvent::HostFuncReturn(_) => "HostFuncReturn",
                RREvent::WasmFuncReturn(_) => "WasmFuncReturn",
                other => panic!("unexpected event in reentrant trace: {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            classes,
            vec![
                "CoreWasmInstantiation",
                "CoreWasmFuncEntry",
                "HostFuncEntry",
                "CoreWasmFuncEntry",
                "WasmFuncReturn",
                "HostFuncReturn",
                "WasmFuncReturn",
            ]
        );
        Ok(())
    }

    #[test]
    fn mismatch_diagnostics_remain_clear() -> Result<()> {
        let event = RREvent::HostFuncEntry(common_events::HostFuncEntryEvent {
            args: RRFuncArgVals {
                bytes: vec![0, 0, 0, 21],
                sizes: vec![4],
            },
        });

        let err = common_events::HostFuncReturnEvent::try_from(event.clone()).unwrap_err();
        assert!(matches!(err, ReplayError::IncorrectEventVariant));
        assert!(format!("{event}").contains("HostFuncEntryEvent"));
        Ok(())
    }

    fn core_host_boundary_module() -> &'static str {
        r#"
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
        "#
    }

    fn measure_disabled(calls: u64, runs: u32) -> Result<Duration> {
        let engine = create_engine(RRConfig::None)?;
        let module = Module::new(&engine, core_host_boundary_module())?;
        let mut linker = Linker::new(&engine);
        linker.func_wrap("env", "nop", || {})?;
        let mut store = Store::new(&engine, ());
        let instance = linker.instantiate(&mut store, &module)?;
        let run = instance.get_typed_func::<u64, ()>(&mut store, "run")?;

        let start = Instant::now();
        for _ in 0..runs {
            run.call(&mut store, calls)?;
        }
        Ok(start.elapsed())
    }

    fn measure_recording(calls: u64, runs: u32) -> Result<Duration> {
        let engine = create_engine(RRConfig::Recording)?;
        let module = Module::new(&engine, core_host_boundary_module())?;
        let mut linker = Linker::new(&engine);
        linker.func_wrap("env", "nop", || {})?;
        let mut store = Store::new(&engine, ());
        store.record(sink(), RecordSettings::default())?;
        let instance = linker.instantiate(&mut store, &module)?;
        let run = instance.get_typed_func::<u64, ()>(&mut store, "run")?;

        let start = Instant::now();
        for _ in 0..runs {
            run.call(&mut store, calls)?;
        }
        Ok(start.elapsed())
    }

    fn record_trace_for_replay(calls: u64) -> Result<Vec<u8>> {
        let engine = create_engine(RRConfig::Recording)?;
        let module = Module::new(&engine, core_host_boundary_module())?;
        let mut linker = Linker::new(&engine);
        linker.func_wrap("env", "nop", || {})?;
        let mut store = Store::new(&engine, ());
        store.record(Cursor::new(Vec::new()), RecordSettings::default())?;
        let instance = linker.instantiate(&mut store, &module)?;
        let run = instance.get_typed_func::<u64, ()>(&mut store, "run")?;
        run.call(&mut store, calls)?;
        Ok(take_trace_cursor(store)?.into_inner())
    }

    fn measure_replay(calls: u64, runs: u32) -> Result<Duration> {
        let trace = record_trace_for_replay(calls)?;
        let engine = create_engine(RRConfig::Replaying)?;
        let module = Module::new(&engine, core_host_boundary_module())?;
        let mut elapsed = Duration::ZERO;

        for _ in 0..runs {
            let mut renv = ReplayEnvironment::new(&engine, ReplaySettings::default());
            renv.add_module(module.clone());
            let mut replay = renv.instantiate_with(
                Cursor::new(trace.clone()),
                |_| Ok(()),
                |linker| {
                    linker.allow_shadowing(true);
                    linker.func_wrap("env", "nop", || -> crate::Result<()> {
                        panic!("host function body should not execute during replay")
                    })?;
                    Ok(())
                },
                |_| Ok(()),
            )?;
            let start = Instant::now();
            replay.run_to_completion()?;
            elapsed += start.elapsed();
        }

        Ok(elapsed)
    }

    #[test]
    #[ignore = "manual performance smoke test"]
    fn perf_report_core_host_boundary() -> Result<()> {
        let profiles = [("micro", 1_000, 200), ("macro", 100_000, 10)];

        for (label, calls, runs) in profiles {
            let disabled = measure_disabled(calls, runs)?;
            let recording = measure_recording(calls, runs)?;
            let replay = measure_replay(calls, runs)?;
            let total_calls = calls as f64 * runs as f64;
            let disabled_ns = disabled.as_nanos() as f64 / total_calls;
            let recording_ns = recording.as_nanos() as f64 / total_calls;
            let replay_ns = replay.as_nanos() as f64 / total_calls;
            let recording_overhead = ((recording_ns / disabled_ns) - 1.0) * 100.0;
            let replay_overhead = ((replay_ns / disabled_ns) - 1.0) * 100.0;

            eprintln!(
                "{label}: disabled={disabled_ns:.2}ns/call recording={recording_ns:.2}ns/call ({recording_overhead:+.1}%) replay={replay_ns:.2}ns/call ({replay_overhead:+.1}%)"
            );
        }

        Ok(())
    }
}
