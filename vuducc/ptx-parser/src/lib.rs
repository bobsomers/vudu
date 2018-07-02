extern crate comment_strip;
#[macro_use]
extern crate nom;

use nom::{digit, is_space, multispace};
use std::str;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    UnknownTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Version {
    pub major: u8,
    pub minor: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Sm30,
    Sm70,
}

impl FromStr for Target {
    type Err = ParseError;

    fn from_str(target: &str) -> Result<Self, Self::Err> {
        match target {
            "sm_30" => Ok(Target::Sm30),
            "sm_70" => Ok(Target::Sm70),
            _ => Err(ParseError::UnknownTarget),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ModuleDirectives {
    pub version: Version,
    pub target: Target,
    pub address_size: u8,
}

named!(u8_literal(&[u8]) -> u8,
    map_res!(map_res!(digit, str::from_utf8), |s: &str| s.parse::<u8>())
);

named!(version_directive(&[u8]) -> Version,
    do_parse!(
        tag!(".version") >>
        take_while!(is_space) >>
        major: u8_literal >>
        char!('.') >>
        minor: u8_literal >>
        (Version { major, minor: minor })
    )
);

named!(target_literal(&[u8]) -> Target,
    map_res!(
        map_res!(
            alt!(tag!("sm_30")
                | tag!("sm_70")
            ),
            str::from_utf8
        ),
        |s: &str| s.parse::<Target>()
    )
);

named!(target_directive(&[u8]) -> Target,
    do_parse!(
        tag!(".target") >>
        take_while!(is_space) >>
        // TODO: Extend this to support debug and platform options
        target: target_literal >>
        (target)
    )
);

named!(address_size_directive(&[u8]) -> u8,
    do_parse!(
        tag!(".address_size") >>
        take_while!(is_space) >>
        address_size: map_res!(map_res!(alt!(tag!("32") | tag!("64")),
                                        str::from_utf8),
                               |s: &str| s.parse::<u8>()) >>
        (address_size)
    )
);

named!(module_directives(&[u8]) -> ModuleDirectives,
    do_parse!(
        version: terminated!(version_directive, multispace) >>
        target: terminated!(target_directive, multispace) >>
        address_size: address_size_directive >>
        (ModuleDirectives { version, target, address_size })
    )
);

fn preprocess(input: &[u8]) -> String {
    let src = String::from_utf8(input.to_vec()).unwrap();
    let clean = comment_strip::strip_comments(src,
                                              comment_strip::CommentStyle::C,
                                              true).unwrap();
    let mut nonempty = clean.lines()
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<&str>>()
                        .join("\n");
    nonempty.push('\n');
    nonempty
}

pub fn parse(input: &[u8]) -> ModuleDirectives {
    // TODO: parser error handling, comment stripping error handling, etc.
    module_directives(preprocess(input).as_bytes()).unwrap().1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_from_str_test() {
        assert_eq!(Target::from_str("sm_30"),
                   Ok(Target::Sm30));
        assert_eq!(Target::from_str("foobar"),
                   Err(ParseError::UnknownTarget));
    }

    #[test]
    fn u8_literal_test() {
        assert_eq!(u8_literal(b"42 "),
            Ok((
                &b" "[..],
                42
            ))
        );
        assert_eq!(u8_literal(b"foo"),
            Err(nom::Err::Error(error_position!(
                &b"foo"[..],
                nom::ErrorKind::Digit
            )))
        );
    }

    #[test]
    fn version_directive_test() {
        assert_eq!(version_directive(b".version 6.0 "),
            Ok((
                &b" "[..],
                Version { major: 6, minor: 0 }
            ))
        );
        assert_eq!(version_directive(b".version 12.42 "),
            Ok((
                &b" "[..],
                Version { major: 12, minor: 42 }
            ))
        );
        assert_eq!(version_directive(b".version 6 "),
            Err(nom::Err::Error(error_position!(
                &b" "[..],
                nom::ErrorKind::Char
            )))
        );
        assert_eq!(version_directive(b".version 6. "),
            Err(nom::Err::Error(error_position!(
                &b" "[..],
                nom::ErrorKind::Digit
            )))
        );
        assert_eq!(version_directive(b".version a.b"),
            Err(nom::Err::Error(error_position!(
                &b"a.b"[..],
                nom::ErrorKind::Digit
            )))
        );
        assert_eq!(version_directive(b".version 6.a"),
            Err(nom::Err::Error(error_position!(
                &b"a"[..],
                nom::ErrorKind::Digit
            )))
        );
    }

    #[test]
    fn target_literal_test() {
        assert_eq!(target_literal(b"sm_30"),
            Ok((
                &b""[..],
                Target::Sm30
            ))
        );
        assert_eq!(target_literal(b"sm_70"),
            Ok((
                &b""[..],
                Target::Sm70
            ))
        );
        assert_eq!(target_literal(b"sm_80"),
            Err(nom::Err::Error(error_position!(
                &b"sm_80"[..],
                nom::ErrorKind::Alt
            )))
        );
    }

    #[test]
    fn target_directive_test() {
        assert_eq!(target_directive(b".target sm_30 "),
            Ok((
                &b" "[..],
                Target::Sm30
            ))
        );
        assert_eq!(target_directive(b".target sm_70 "),
            Ok((
                &b" "[..],
                Target::Sm70
            ))
        );
        assert_eq!(target_directive(b".target foobar"),
            Err(nom::Err::Error(error_position!(
                &b"foobar"[..],
                nom::ErrorKind::Alt
            )))
        );
    }

    #[test]
    fn address_size_directive_test() {
        assert_eq!(address_size_directive(b".address_size 32 "),
            Ok((
                &b" "[..],
                32u8
            ))
        );
        assert_eq!(address_size_directive(b".address_size 64 "),
            Ok((
                &b" "[..],
                64u8
            ))
        );
        assert_eq!(address_size_directive(b".address_size 128"),
            Err(nom::Err::Error(error_position!(
                &b"128"[..],
                nom::ErrorKind::Alt
            )))
        );
        assert_eq!(address_size_directive(b".address_size foo"),
            Err(nom::Err::Error(error_position!(
                &b"foo"[..],
                nom::ErrorKind::Alt
            )))
        );
    }

    #[test]
    fn module_directives_test() {
        let src = include_bytes!("test_ptx/module_directives_test.ptx");
        assert_eq!(module_directives(src),
            Ok((
                &b"\n"[..],
                ModuleDirectives {
                    version: Version { major: 6, minor: 0 },
                    target: Target::Sm30,
                    address_size: 64u8
                }
            ))
        );

        let src = preprocess(include_bytes!("test_ptx/messy_source.ptx"));
        assert_eq!(module_directives(src.as_bytes()),
            Ok((
                &b"\n"[..],
                ModuleDirectives {
                    version: Version { major: 6, minor: 2 },
                    target: Target::Sm70,
                    address_size: 32u8
                }
            ))
        );
    }

    #[test]
    fn preprocess_test() {
        let src = preprocess(include_bytes!("test_ptx/messy_source.ptx"));
        assert_eq!(src, ".version 6.2\n.target sm_70\n.address_size 32\n");
    }
}
