use std::env;

mod lang;

fn main() {
    let args = env::args().collect::<Vec<String>>();
    let source = {
        if args.len() > 1 {
            let filepath = args.get(1).unwrap();
            std::fs::read_to_string(filepath).unwrap()
        } else {
            r#"
                fn main(args: [string]) -> void {
                    print("Hello, world!\n");
                }
            "#
            .to_string()
        }
    };

    lang::run_interpreter(&source, &["test-program"]);
}
