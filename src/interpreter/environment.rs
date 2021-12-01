use super::parser::RootFunctionDeclaration;
use std::{cell::RefCell, collections::HashMap, rc::Rc};

#[derive(Debug, PartialEq)]
pub struct Env<T: Clone> {
    symbols: RefCell<HashMap<String, ValueInfo<T>>>,
    parent_env: Option<Rc<Env<T>>>,
}

#[derive(Debug, PartialEq)]
struct ValueInfo<T: Clone> {
    immutable: bool,
    value: T,
}

#[derive(Debug, PartialEq)]
pub enum EnvError {
    SymbolNotFound(String),
    CannotAssignToImmutableValue(String),
}

impl<T: Clone> Env<T> {
    fn base_env() -> Rc<Env<T>> {
        let e = Env {
            symbols: RefCell::new(HashMap::new()),
            parent_env: None,
        };
        // TODO: Add native functions here
        Rc::new(e)
    }

    pub fn root_env(_root_function_declarations: &[RootFunctionDeclaration]) -> Rc<Env<T>> {
        // TODO: Fill the root env with the outmost function declarations
        Env::base_env()
    }

    pub fn create_child(parent: &Rc<Env<T>>) -> Rc<Env<T>> {
        Rc::new(Env {
            symbols: RefCell::new(HashMap::new()),
            parent_env: Some(Rc::clone(parent)),
        })
    }

    pub fn declare(&self, symbol: &str, value: T, immutable: bool) {
        self.symbols
            .borrow_mut()
            .insert(symbol.to_string(), ValueInfo { value, immutable });
    }

    pub fn assign(&self, symbol: &str, value: T) -> Result<(), EnvError> {
        match (
            self.symbols.borrow_mut().get_mut(symbol),
            self.parent_env.as_ref(),
        ) {
            (None, None) => Err(EnvError::SymbolNotFound(symbol.to_string())),
            (None, Some(parent)) => parent.assign(symbol, value),
            (Some(value_info), _) if value_info.immutable => {
                Err(EnvError::CannotAssignToImmutableValue(symbol.to_string()))
            }
            (Some(value_info), _) => {
                value_info.value = value;
                Ok(())
            }
        }
    }

    pub fn lookup(&self, symbol: &str) -> Option<T> {
        // FIXME: It's unfortunate having to clone every value we lookup

        self.symbols
            .borrow()
            .get(symbol)
            .map(|value_info| value_info.value.clone())
            .or_else(|| match &self.parent_env {
                Some(parent) => parent.lookup(symbol),
                None => None,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declare_and_lookup() {
        let env = Env::base_env();
        let _ = env.declare("myval", 1, false);
        assert_eq!(env.lookup("myval"), Some(1));
    }

    #[test]
    fn lookup_parent() {
        let parent = Env::base_env();
        let _ = parent.declare("myval", 1, false);
        let child = Env::create_child(&parent);
        assert_eq!(child.lookup("myval"), Some(1));
    }

    #[test]
    fn assign_from_child_to_parent() {
        let parent = Env::base_env();
        let _ = parent.declare("myval", 1, false);
        let child = Env::create_child(&parent);
        let _ = child.assign("myval", 2);
        assert_eq!(parent.lookup("myval"), Some(2));
    }

    #[test]
    fn declare_variable_in_child() {
        let parent = Env::base_env();
        let child = Env::create_child(&parent);
        let _ = child.declare("myval", 1, false);
        assert_eq!(parent.lookup("myval"), None);
    }

    #[test]
    fn child_symbol_shadows_parents() {
        let parent = Env::base_env();
        let _ = parent.declare("myval", 1, true);
        let child = Env::create_child(&parent);
        let _ = child.declare("myval", 2, false);
        assert_eq!(parent.lookup("myval"), Some(1));
        assert_eq!(child.lookup("myval"), Some(2));
    }

    #[test]
    fn assign_to_const_fails() {
        let parent = Env::base_env();
        let _ = parent.declare("myval", 1, true);
        assert_eq!(
            parent.assign("myval", 2),
            Err(EnvError::CannotAssignToImmutableValue("myval".to_string()))
        );
    }

    #[test]
    fn assign_to_undeclared_fails() {
        let parent = Env::base_env();
        assert_eq!(
            parent.assign("undeclared", 1),
            Err(EnvError::SymbolNotFound("undeclared".to_string()))
        );
    }
}
