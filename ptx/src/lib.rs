mod lexer;

pub fn parse(src: &str) {
    let _tokens = lexer::lex(src);
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
