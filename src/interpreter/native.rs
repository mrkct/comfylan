use lazy_static::lazy_static;

use crate::interpreter::{ast::ImmediateValue, environment::Env, typechecking::Type};
use std::{cell::RefCell, fmt, rc::Rc};

use super::evaluator::EvaluationError;
use rand::prelude::*;

lazy_static! {
    static ref NATIVE_FUNCTIONS: [(&'static str, NativeFunction); 9] = [
        (
            "print",
            NativeFunction {
                tag: 0,
                signature: Type::Closure(
                    vec![Type::VarArgs(Box::new(Type::Any))],
                    Box::new(Type::Void)
                ),
                callback: native_print
            }
        ),
        (
            "len",
            NativeFunction {
                tag: 1,
                signature: Type::Closure(
                    vec![Type::Array(Box::new(Type::Any))],
                    Box::new(Type::Integer)
                ),
                callback: native_array_len
            }
        ),
        (
            "insert",
            NativeFunction {
                tag: 2,
                signature: Type::Closure(
                    vec![Type::Array(Box::new(Type::Any)), Type::Integer, Type::Any],
                    Box::new(Type::Any)
                ),
                callback: native_array_insert
            }
        ),
        (
            "remove",
            NativeFunction {
                tag: 3,
                signature: Type::Closure(
                    vec![Type::Array(Box::new(Type::Any)), Type::Integer],
                    Box::new(Type::Void)
                ),
                callback: native_array_remove
            }
        ),
        (
            "random",
            NativeFunction {
                tag: 4,
                signature: Type::Closure(
                    vec![Type::Integer, Type::Integer],
                    Box::new(Type::Integer)
                ),
                callback: native_random
            }
        ),
        (
            "delay",
            NativeFunction {
                tag: 5,
                signature: Type::Closure(vec![Type::Integer], Box::new(Type::Void)),
                callback: native_delay
            }
        ),
        (
            "open_window",
            NativeFunction {
                tag: 6,
                signature: Type::Closure(
                    vec![Type::Integer, Type::Integer, Type::String],
                    Box::new(Type::Void)
                ),
                callback: native_open_window
            }
        ),
        (
            "refresh_screen",
            NativeFunction {
                tag: 7,
                signature: Type::Closure(vec![], Box::new(Type::Void)),
                callback: native_refresh_screen
            }
        ),
        (
            "exit",
            NativeFunction {
                tag: 8,
                signature: Type::Closure(vec![], Box::new(Type::Void)),
                callback: native_exit
            }
        ),
    ];
}

#[derive(Clone)]
pub struct NativeFunction {
    tag: i32,
    pub signature: Type,
    pub callback: fn(
        &mut dyn GameEngineSubsystem,
        Vec<ImmediateValue>,
    ) -> Result<ImmediateValue, EvaluationError>,
}

impl fmt::Debug for NativeFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeFunction")
            .field("tag", &self.tag)
            .field("signature", &self.signature)
            .finish()
    }
}

impl PartialEq for NativeFunction {
    fn eq(&self, other: &Self) -> bool {
        self.tag == other.tag
    }
}

pub trait GameEngineSubsystem {
    fn open_window(&mut self, w: u32, h: u32, title: &str) -> Result<(), String>;
    fn refresh_screen(&mut self) -> Result<(), String>;
}

pub fn fill_values_env_with_native_functions(env: &Rc<Env<ImmediateValue>>) {
    for (name, native_function) in NATIVE_FUNCTIONS.iter() {
        env.declare(
            name,
            ImmediateValue::NativeFunction(native_function.clone()),
        );
    }
}

pub fn fill_type_env_with_native_functions(env: &Rc<Env<Type>>) {
    for (name, NativeFunction { signature, .. }) in NATIVE_FUNCTIONS.iter() {
        env.declare(name, signature.clone());
    }
}

pub fn get_default_system_game_engine_subsystem() -> impl GameEngineSubsystem {
    sdl_subsystem::SdlSubsystem::new()
}

fn print_immediate_value(v: &ImmediateValue) {
    match v {
        ImmediateValue::Integer(x) => print!("{}", x),
        ImmediateValue::FloatingPoint(x) => print!("{}", x),
        ImmediateValue::String(s) => print!("{}", s),
        ImmediateValue::Boolean(b) => print!("{}", b),
        ImmediateValue::Array(_, a) => {
            print!("[");
            let a_borrow = a.borrow();
            let mut iter = a_borrow.iter();
            if let Some(v) = iter.next() {
                print_immediate_value(v);
            }
            for v in iter {
                print!(", ");
                print_immediate_value(v);
            }
            print!("]");
        }
        ImmediateValue::Closure(ftype, _, _, _) => print!("[@Closure {:?}]", ftype),
        ImmediateValue::NativeFunction(function) => print!("[@NativeFunction {:?}]", function),
        ImmediateValue::Void => print!("[Void]"),
        ImmediateValue::Struct(_, fields) => {
            print!("{{");
            let borrow = fields.borrow();
            let mut iter = borrow.iter();
            if let Some((name, val)) = iter.next() {
                print!("{}=", name);
                print_immediate_value(val);
            }
            for (name, val) in iter {
                print!(", {}=", name);
                print_immediate_value(val);
            }
            print!("}}");
        }
    }
}

fn native_print(
    _: &mut dyn GameEngineSubsystem,
    args: Vec<ImmediateValue>,
) -> Result<ImmediateValue, EvaluationError> {
    for arg in args {
        print_immediate_value(&arg);
    }
    Ok(ImmediateValue::Void)
}

fn native_array_len(
    _: &mut dyn GameEngineSubsystem,
    args: Vec<ImmediateValue>,
) -> Result<ImmediateValue, EvaluationError> {
    match args.get(0) {
        Some(ImmediateValue::Array(_, array)) => {
            Ok(ImmediateValue::Integer(array.borrow().len() as i64))
        },
        _ => panic!("Typechecker failed! Native function 'len' was called with an argument that is not an array")
    }
}

fn validate_array_index(
    array: &Rc<RefCell<Vec<ImmediateValue>>>,
    index: i64,
) -> Result<usize, EvaluationError> {
    let array_len = array.borrow().len();
    if index < 0 {
        return Err(EvaluationError::ArrayIndexOutOfBounds(array_len, index));
    }

    let usize_index: usize = index
        .try_into()
        .map_err(|_| EvaluationError::ArrayIndexOutOfBounds(array_len, index))?;
    if usize_index > array_len {
        return Err(EvaluationError::ArrayIndexOutOfBounds(array_len, index));
    }
    Ok(usize_index)
}

fn native_array_insert(
    _: &mut dyn GameEngineSubsystem,
    args: Vec<ImmediateValue>,
) -> Result<ImmediateValue, EvaluationError> {
    match (args.get(0), args.get(1), args.get(2)) {
        (Some(ImmediateValue::Array(_, array)), Some(ImmediateValue::Integer(index)), Some(v)) => {
            let i = validate_array_index(array, *index)?;
            array.borrow_mut().insert(i, v.clone())
        }
        _ => panic!("Typechecker failed! Native function 'insert' was called with bad arguments"),
    }
    Ok(ImmediateValue::Void)
}

fn native_array_remove(
    _: &mut dyn GameEngineSubsystem,
    args: Vec<ImmediateValue>,
) -> Result<ImmediateValue, EvaluationError> {
    match (args.get(0), args.get(1)) {
        (Some(ImmediateValue::Array(_, array)), Some(ImmediateValue::Integer(index))) => {
            let i = validate_array_index(array, *index)?;
            Ok(array.borrow_mut().remove(i))
        }
        _ => panic!("Typechecker failed! Native function 'remove' was called with bad arguments"),
    }
}

fn native_random(
    _: &mut dyn GameEngineSubsystem,
    args: Vec<ImmediateValue>,
) -> Result<ImmediateValue, EvaluationError> {
    match (args.get(0), args.get(1)) {
        (Some(ImmediateValue::Integer(x1)), Some(ImmediateValue::Integer(x2))) => {
            let low = *x1.min(x2);
            let high = *x1.max(x2);
            Ok(ImmediateValue::Integer(
                rand::thread_rng().gen_range(low..=high),
            ))
        }
        _ => panic!("typechecker failed?"),
    }
}

fn native_delay(
    _: &mut dyn GameEngineSubsystem,
    args: Vec<ImmediateValue>,
) -> Result<ImmediateValue, EvaluationError> {
    match args.get(0) {
        Some(ImmediateValue::Integer(ms)) => {
            let ms = *ms;
            if ms < 0 {
                Err(EvaluationError::NativeSpecific(format!(
                    "Delay was called with negative value ({})",
                    ms
                )))
            } else {
                std::thread::sleep(std::time::Duration::from_millis(ms.try_into().unwrap()));
                Ok(ImmediateValue::Void)
            }
        }
        _ => panic!("typechecker failed?"),
    }
}

fn native_exit(
    _: &mut dyn GameEngineSubsystem,
    _: Vec<ImmediateValue>,
) -> Result<ImmediateValue, EvaluationError> {
    std::process::exit(0);
}

fn native_open_window(
    subsystem: &mut dyn GameEngineSubsystem,
    args: Vec<ImmediateValue>,
) -> Result<ImmediateValue, EvaluationError> {
    match (args.get(0), args.get(1), args.get(2)) {
        (
            Some(ImmediateValue::Integer(w)),
            Some(ImmediateValue::Integer(h)),
            Some(ImmediateValue::String(title)),
        ) => {
            let w: u32 = (*w).try_into().unwrap();
            let h: u32 = (*h).try_into().unwrap();

            subsystem
                .open_window(w, h, title)
                .map(|_| ImmediateValue::Void)
                .map_err(EvaluationError::NativeSpecific)
        }
        _ => panic!("typechecker failed?"),
    }
}

fn native_refresh_screen(
    subsystem: &mut dyn GameEngineSubsystem,
    _: Vec<ImmediateValue>,
) -> Result<ImmediateValue, EvaluationError> {
    subsystem
        .refresh_screen()
        .map(|_| ImmediateValue::Void)
        .map_err(EvaluationError::NativeSpecific)
}

mod sdl_subsystem {

    extern crate sdl2;

    use super::GameEngineSubsystem;
    use sdl2::{render::Canvas, video::Window, EventPump, Sdl, VideoSubsystem};

    pub struct SdlSubsystem {
        sdl_context: Sdl,
        video_subsystem: VideoSubsystem,
        canvas: Option<Canvas<Window>>,
        event_pump: Option<EventPump>,
    }

    impl GameEngineSubsystem for SdlSubsystem {
        fn open_window(&mut self, width: u32, height: u32, title: &str) -> Result<(), String> {
            let window = self
                .video_subsystem
                .window(title, width, height)
                .position_centered()
                .opengl()
                .build()
                .map_err(|e| e.to_string())?;
            let canvas = window.into_canvas().build().map_err(|e| e.to_string())?;
            self.canvas = Some(canvas);

            Ok(())
        }

        fn refresh_screen(&mut self) -> Result<(), String> {
            match &mut self.canvas {
                Some(canvas) => {
                    canvas.present();
                    Ok(())
                }
                None => Err(String::from(
                    "Cannot refresh screen because the window is not currently open",
                )),
            }
        }
    }

    impl SdlSubsystem {
        pub fn new() -> SdlSubsystem {
            let sdl_context = sdl2::init().unwrap();
            let video_subsystem = sdl_context.video().unwrap();
            SdlSubsystem {
                sdl_context,
                video_subsystem,
                canvas: None,
                event_pump: None,
            }
        }
    }
}
