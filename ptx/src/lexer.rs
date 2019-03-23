pub enum Token {
//    Directive,
//    Instruction,
//    Label,
//    Separator,
//    Literal,
}

// TODO: Add source locations?
#[derive(Debug, PartialEq)]
pub enum LexError {
    MissingNewline,
    MissingBlockCommentClose,
}

// TODO: Strip comments from token stream.
pub fn lex(src: &str) -> Result<Vec<Token>, LexError> {
    // Trim leading comments and whitespace.
    let src = trim_comment_start(src)?;
    let _src = src.trim_start();

    // TODO
    Ok(vec![])
}

/// Removes any number of line or block comments from the start of `src`,
/// returning a slice into `src` with the comments removed.
fn trim_comment_start(src: &str) -> Result<&str, LexError> {
    loop {
        let stable = src;
        let src = trim_line_comment_start(src.trim_start())?;
        let src = trim_block_comment_start(src.trim_start())?;
        if src == stable {
            break Ok(src);
        }
    }
}

/// Removes a single line-style comment from the start of `src`, returning a
/// slice into `src` with the comment removed.
///
/// Expects that `src` has been trimmed of leading whitespace.
fn trim_line_comment_start(src: &str) -> Result<&str, LexError> {
    if src.starts_with("//") {
        match src.find('\n') {
            Some(n) => Ok(&src[n + 1..]),
            None => Err(LexError::MissingNewline),
        }
    } else {
        Ok(src)
    }
}

/// Removes a single block-style comment from the start of `src`, returning a
/// slice into `src` with the comment removed.
///
/// Expects that `src` has been trimmed of leading whitespace.
fn trim_block_comment_start(src: &str) -> Result<&str, LexError> {
    if src.starts_with("/*") {
        match src.find("*/") {
            Some(n) => Ok(&src[n + 2..]),
            None => Err(LexError::MissingBlockCommentClose),
        }
    } else {
        Ok(src)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_line_comment_start_test() {
        let src = "// This is a comment\nThis is not.";
        assert_eq!(trim_line_comment_start(src).unwrap(), "This is not.");

        let src = "////////////\nThis is not.";
        assert_eq!(trim_line_comment_start(src).unwrap(), "This is not.");

        let src = "/";
        assert_eq!(trim_line_comment_start(src).unwrap(), "/");

        let src = "// This is a comment with no newline.";
        assert_eq!(trim_line_comment_start(src).unwrap_err(), LexError::MissingNewline);

        let src = "// This is a final comment.\n";
        assert_eq!(trim_line_comment_start(src).unwrap(), "");
    }

    #[test]
    fn trim_block_comment_start_test() {
        let src = "/* This is a comment. */This is not.";
        assert_eq!(trim_block_comment_start(src).unwrap(), "This is not.");

        let src = "/**************/This is not.";
        assert_eq!(trim_block_comment_start(src).unwrap(), "This is not.");

        let src = "/";
        assert_eq!(trim_block_comment_start(src).unwrap(), "/");

        let src = "/* This comment has no end.";
        assert_eq!(trim_block_comment_start(src).unwrap_err(), LexError::MissingBlockCommentClose);

        let src = "/* This is a final comment. */";
        assert_eq!(trim_block_comment_start(src).unwrap(), "");
    }

    #[test]
    fn trim_comment_start_test() {
        // TODO
        assert_eq!(2 + 2, 4);
    }
}
