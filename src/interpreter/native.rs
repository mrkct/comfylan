use lazy_static::lazy_static;

use crate::interpreter::{ast::ImmediateValue, environment::Env, typechecking::Type};
use std::{cell::RefCell, collections::HashMap, fmt, rc::Rc};

use super::evaluator::EvaluationError;
use rand::prelude::*;

#[cfg(test)]
use mockall::automock;

lazy_static! {
    static ref NATIVE_FUNCTIONS: [(&'static str, NativeFunction); 15] = [
        (
            "rect",
            NativeFunction {
                tag: 0,
                signature: Type::Closure(
                    vec![Type::Integer, Type::Integer, Type::Integer, Type::Integer],
                    Box::new(Type::TypeReference("Rect".to_string()))
                ),
                callback: native_rect
            }
        ),
        (
            "rgb",
            NativeFunction {
                tag: 0,
                signature: Type::Closure(
                    vec![Type::Integer, Type::Integer, Type::Integer],
                    Box::new(Type::TypeReference("Rgb".to_string()))
                ),
                callback: native_rgb
            }
        ),
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
        (
            "draw_rectangle",
            NativeFunction {
                tag: 9,
                signature: Type::Closure(
                    vec![
                        Type::TypeReference(String::from("Rect")),
                        Type::TypeReference(String::from("Rgb")),
                        Type::Boolean
                    ],
                    Box::new(Type::Void)
                ),
                callback: native_draw_rectangle
            }
        ),
        (
            "mouse_x",
            NativeFunction {
                tag: 10,
                signature: Type::Closure(vec![], Box::new(Type::Integer)),
                callback: native_mouse_x
            }
        ),
        (
            "mouse_y",
            NativeFunction {
                tag: 11,
                signature: Type::Closure(vec![], Box::new(Type::Integer)),
                callback: native_mouse_y
            }
        ),
        (
            "mouse_btn_down",
            NativeFunction {
                tag: 12,
                signature: Type::Closure(vec![Type::Integer], Box::new(Type::Boolean)),
                callback: native_mouse_btn_down
            }
        ),
    ];
    static ref NATIVE_TYPES: [(&'static str, Type); 2] = [
        (
            "Rgb",
            Type::Struct(
                "Rgb".to_string(),
                HashMap::from([
                    ("r".to_string(), Type::Integer),
                    ("g".to_string(), Type::Integer),
                    ("b".to_string(), Type::Integer),
                ])
            )
        ),
        (
            "Rect",
            Type::Struct(
                "Rect".to_string(),
                HashMap::from([
                    ("x".to_string(), Type::Integer),
                    ("y".to_string(), Type::Integer),
                    ("w".to_string(), Type::Integer),
                    ("h".to_string(), Type::Integer),
                ])
            )
        )
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

#[cfg_attr(test, automock)]
pub trait GameEngineSubsystem {
    fn open_window(&mut self, w: u32, h: u32, title: &str) -> Result<(), String>;
    fn refresh_screen(&mut self) -> Result<(), String>;
    fn draw_rectangle(
        &mut self,
        x: i32,
        y: i32,
        w: u32,
        h: u32,
        r: u8,
        g: u8,
        b: u8,
        fill: bool,
    ) -> Result<(), String>;

    fn mouse_state(&mut self) -> Result<MouseState, String>;
}

#[derive(Debug, PartialEq, Clone)]
pub struct MouseState {
    pub x: i64,
    pub y: i64,
    pub left_button_down: bool,
    pub middle_button_down: bool,
    pub right_button_down: bool,
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

pub fn declare_native_types(user_types: &mut HashMap<String, Type>) {
    for (name, t) in NATIVE_TYPES.iter() {
        user_types.insert(name.to_string(), t.clone());
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

fn native_rect(
    _: &mut dyn GameEngineSubsystem,
    args: Vec<ImmediateValue>,
) -> Result<ImmediateValue, EvaluationError> {
    let x = unwrap_expecting_integer(args.get(0));
    let y = unwrap_expecting_integer(args.get(1));
    let w = unwrap_expecting_integer(args.get(2));
    let h = unwrap_expecting_integer(args.get(3));

    Ok(ImmediateValue::Struct(
        Type::TypeReference("Rect".to_string()),
        Rc::new(RefCell::new(HashMap::from([
            ("x".to_string(), ImmediateValue::Integer(x)),
            ("y".to_string(), ImmediateValue::Integer(y)),
            ("w".to_string(), ImmediateValue::Integer(w)),
            ("h".to_string(), ImmediateValue::Integer(h)),
        ]))),
    ))
}

fn native_rgb(
    _: &mut dyn GameEngineSubsystem,
    args: Vec<ImmediateValue>,
) -> Result<ImmediateValue, EvaluationError> {
    let r = unwrap_expecting_integer(args.get(0));
    let g = unwrap_expecting_integer(args.get(1));
    let b = unwrap_expecting_integer(args.get(2));

    Ok(ImmediateValue::Struct(
        Type::TypeReference("Rgb".to_string()),
        Rc::new(RefCell::new(HashMap::from([
            ("r".to_string(), ImmediateValue::Integer(r)),
            ("g".to_string(), ImmediateValue::Integer(g)),
            ("b".to_string(), ImmediateValue::Integer(b)),
        ]))),
    ))
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

fn unwrap_expecting_struct(
    x: Option<&ImmediateValue>,
) -> &Rc<RefCell<HashMap<String, ImmediateValue>>> {
    match x {
        Some(ImmediateValue::Struct(_, r)) => r,
        _ => panic!("Expected to unwrap an ImmediateValue::Struct"),
    }
}

fn unwrap_expecting_bool(x: Option<&ImmediateValue>) -> bool {
    match x {
        Some(ImmediateValue::Boolean(b)) => *b,
        _ => panic!("Expected to unwrap an ImmediateValue::Boolean"),
    }
}

fn unwrap_expecting_integer(x: Option<&ImmediateValue>) -> i64 {
    match x {
        Some(ImmediateValue::Integer(i)) => *i,
        t => panic!(
            "Expected to unwrap an ImmediateValue::Integer, got {:?} instead",
            t
        ),
    }
}

fn unwrap_expecting_rect(
    rect: Option<&ImmediateValue>,
) -> Result<(i32, i32, u32, u32), EvaluationError> {
    let rect = unwrap_expecting_struct(rect);
    let borrow = rect.borrow();
    let x: i32 = unwrap_expecting_integer(borrow.get("x"))
        .try_into()
        .map_err(|_| {
            EvaluationError::NativeSpecific(String::from("x coordinate is too big for the window"))
        })?;
    let y: i32 = unwrap_expecting_integer(borrow.get("y"))
        .try_into()
        .map_err(|_| {
            EvaluationError::NativeSpecific(String::from("y coordinate is too big for the window"))
        })?;
    let w: u32 = unwrap_expecting_integer(borrow.get("w"))
        .try_into()
        .map_err(|_| {
            EvaluationError::NativeSpecific(String::from("Width must be a positive value"))
        })?;
    let h: u32 = unwrap_expecting_integer(borrow.get("h"))
        .try_into()
        .map_err(|_| {
            EvaluationError::NativeSpecific(String::from("Height must be a positive value"))
        })?;
    Ok((x, y, w, h))
}

fn unwrap_expecting_rgb(rgb: Option<&ImmediateValue>) -> Result<(u8, u8, u8), EvaluationError> {
    let rgb = unwrap_expecting_struct(rgb);
    let borrow = rgb.borrow();
    let r: u8 = unwrap_expecting_integer(borrow.get("r"))
        .try_into()
        .map_err(|_| {
            EvaluationError::NativeSpecific(String::from("rgb values must be between 0 and 255"))
        })?;
    let g: u8 = unwrap_expecting_integer(borrow.get("g"))
        .try_into()
        .map_err(|_| {
            EvaluationError::NativeSpecific(String::from("rgb values must be between 0 and 255"))
        })?;
    let b: u8 = unwrap_expecting_integer(borrow.get("b"))
        .try_into()
        .map_err(|_| {
            EvaluationError::NativeSpecific(String::from("rgb values must be between 0 and 255"))
        })?;
    Ok((r, g, b))
}

fn native_draw_rectangle(
    subsystem: &mut dyn GameEngineSubsystem,
    args: Vec<ImmediateValue>,
) -> Result<ImmediateValue, EvaluationError> {
    let (x, y, w, h) = unwrap_expecting_rect(args.get(0))?;
    let (r, g, b) = unwrap_expecting_rgb(args.get(1))?;
    let fill = unwrap_expecting_bool(args.get(2));

    subsystem
        .draw_rectangle(x, y, w, h, r, g, b, fill)
        .map(|_| ImmediateValue::Void)
        .map_err(EvaluationError::NativeSpecific)
}

fn native_mouse_x(
    subsystem: &mut dyn GameEngineSubsystem,
    _: Vec<ImmediateValue>,
) -> Result<ImmediateValue, EvaluationError> {
    subsystem
        .mouse_state()
        .map(|state| ImmediateValue::Integer(state.x))
        .map_err(EvaluationError::NativeSpecific)
}

fn native_mouse_y(
    subsystem: &mut dyn GameEngineSubsystem,
    _: Vec<ImmediateValue>,
) -> Result<ImmediateValue, EvaluationError> {
    subsystem
        .mouse_state()
        .map(|state| ImmediateValue::Integer(state.y))
        .map_err(EvaluationError::NativeSpecific)
}

fn get_mouse_button_status(
    subsystem: &mut dyn GameEngineSubsystem,
    mouse_button_number: i64,
) -> Result<bool, EvaluationError> {
    subsystem
        .mouse_state()
        .map_err(EvaluationError::NativeSpecific)
        .and_then(|state| {
            match mouse_button_number {
                0 => Ok(state.left_button_down),
                1 => Ok(state.middle_button_down),
                2 => Ok(state.right_button_down),
                _ => Err(EvaluationError::NativeSpecific(String::from("Not a valid mouse button number. Use the BTN_LEFT, BTN_MIDDLE or BTN_RIGHT constants!")))
            }
        })
}

fn native_mouse_btn_down(
    subsystem: &mut dyn GameEngineSubsystem,
    args: Vec<ImmediateValue>,
) -> Result<ImmediateValue, EvaluationError> {
    let btn = unwrap_expecting_integer(args.get(0));

    get_mouse_button_status(subsystem, btn).map(ImmediateValue::Boolean)
}

mod sdl_subsystem {

    extern crate sdl2;

    use super::*;
    use sdl2::{event::Event, keyboard::Keycode, pixels::Color, rect::Rect};
    use std::{
        collections::HashMap,
        sync::{mpsc, Arc, RwLock},
        thread,
    };

    pub struct SdlSubsystem {
        tx: mpsc::Sender<Action>,
        rx: mpsc::Receiver<Result<(), String>>,
        input: Arc<InputStatus>,
    }

    #[derive(Debug)]
    enum Action {
        OpenWindow(u32, u32, String),
        Refresh,
        DrawRectangle((i32, i32, u32, u32), (u8, u8, u8), bool),
    }

    struct InputStatus {
        pub keycodes: RwLock<HashMap<Keycode, bool>>,
        pub mouse: RwLock<MouseState>,
    }

    impl GameEngineSubsystem for SdlSubsystem {
        fn open_window(&mut self, width: u32, height: u32, title: &str) -> Result<(), String> {
            self.send_and_wait(Action::OpenWindow(width, height, String::from(title)))
        }

        fn refresh_screen(&mut self) -> Result<(), String> {
            self.send_and_wait(Action::Refresh)
        }

        fn draw_rectangle(
            &mut self,
            x: i32,
            y: i32,
            w: u32,
            h: u32,
            r: u8,
            g: u8,
            b: u8,
            fill: bool,
        ) -> Result<(), String> {
            self.send_and_wait(Action::DrawRectangle((x, y, w, h), (r, g, b), fill))
        }

        fn mouse_state(&mut self) -> Result<MouseState, String> {
            Ok(self.input.mouse.read().unwrap().clone())
        }
    }

    impl SdlSubsystem {
        fn send_and_wait(&mut self, action: Action) -> Result<(), String> {
            self.tx.send(action).unwrap();
            self.rx.recv().unwrap()
        }

        pub fn new() -> SdlSubsystem {
            let (interpreter_tx, sdl_rx) = mpsc::channel();
            let (sdl_tx, interpreter_rx) = mpsc::channel();

            let input = Arc::new(InputStatus {
                keycodes: RwLock::new(HashMap::new()),
                mouse: RwLock::new(MouseState {
                    x: 0,
                    y: 0,
                    left_button_down: false,
                    middle_button_down: false,
                    right_button_down: false,
                }),
            });

            let thread_input = Arc::clone(&input);
            thread::spawn(move || {
                let sdl_context = sdl2::init().unwrap();
                let video_subsystem = sdl_context.video().unwrap();
                let mut event_pump = sdl_context.event_pump().unwrap();
                let mut canvas = None;

                let (tx, rx) = (sdl_tx, sdl_rx);
                let input = thread_input;
                loop {
                    for msg in rx.try_iter() {
                        let result = {
                            match msg {
                                Action::OpenWindow(width, height, title) if canvas.is_none() => {
                                    video_subsystem
                                        .window(&title, width, height)
                                        .position_centered()
                                        .opengl()
                                        .build()
                                        .map_err(|e| e.to_string())
                                        .and_then(|window| {
                                            window
                                                .into_canvas()
                                                .build()
                                                .map_err(|e| e.to_string())
                                                .map(|c| {
                                                    canvas = Some(c);
                                                    ()
                                                })
                                        })
                                }
                                Action::OpenWindow(..) => {
                                    Err(String::from("You already have opened a window!"))
                                }
                                Action::Refresh => canvas
                                    .as_mut()
                                    .ok_or_else(|| {
                                        String::from("You can't refresh a non-existing window|")
                                    })
                                    .map(|canvas| {
                                        canvas.present();
                                        ()
                                    }),
                                Action::DrawRectangle((x, y, w, h), (r, g, b), fill) => canvas
                                    .as_mut()
                                    .ok_or_else(|| {
                                        String::from("You can't draw before opening a window!")
                                    })
                                    .map(|canvas| {
                                        canvas.set_draw_color(Color::RGB(r, g, b));
                                        let rect = Rect::new(x, y, w, h);
                                        if fill {
                                            canvas.fill_rect(rect).unwrap();
                                        } else {
                                            canvas.draw_rect(rect).unwrap();
                                        }
                                        ()
                                    }),
                            }
                        };
                        tx.send(result).unwrap();
                    }

                    for event in event_pump.poll_iter() {
                        match event {
                            Event::Quit { .. }
                            | Event::KeyDown {
                                keycode: Some(Keycode::Escape),
                                ..
                            } => std::process::exit(0),
                            _ => { /* TODO: Handle... */ }
                        }
                    }
                    {
                        let mouse_state = event_pump.mouse_state();
                        let mut mouse_lock = input.mouse.write().unwrap();
                        mouse_lock.x = mouse_state.x() as i64;
                        mouse_lock.y = mouse_state.y() as i64;
                        mouse_lock.left_button_down = mouse_state.left();
                        mouse_lock.middle_button_down = mouse_state.middle();
                        mouse_lock.right_button_down = mouse_state.right();
                    }
                }
            });

            SdlSubsystem {
                tx: interpreter_tx,
                rx: interpreter_rx,
                input,
            }
        }
    }
}
