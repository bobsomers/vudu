extern crate comment_strip;
#[macro_use]
extern crate nom;

use nom::{digit, is_alphabetic, multispace};
use std::str;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    UnknownTarget,
    UnknownVariableType,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableType {
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
    B8,
    B16,
    B32,
    B64,
}

impl FromStr for VariableType {
    type Err = ParseError;

    fn from_str(target: &str) -> Result<Self, Self::Err> {
        match target {
            ".s8" => Ok(VariableType::S8),
            ".s16" => Ok(VariableType::S16),
            ".s32" => Ok(VariableType::S32),
            ".s64" => Ok(VariableType::S64),
            ".u8" => Ok(VariableType::U8),
            ".u16" => Ok(VariableType::U16),
            ".u32" => Ok(VariableType::U32),
            ".u64" => Ok(VariableType::U64),
            ".f16" => Ok(VariableType::F16),
            ".f32" => Ok(VariableType::F32),
            ".f64" => Ok(VariableType::F64),
            ".b8" => Ok(VariableType::B8),
            ".b16" => Ok(VariableType::B16),
            ".b32" => Ok(VariableType::B32),
            ".b64" => Ok(VariableType::B64),
            _ => Err(ParseError::UnknownVariableType),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Param {
    pub var_type: VariableType,
    pub name: String,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub param_list: Vec<Param>,
    pub kernel_body: String,
    pub visible: bool,
}

impl Entry {
    fn new(name: String, param_list: Vec<Param>, kernel_body: String) -> Self {
        Entry {
            name: name,
            param_list: param_list,
            kernel_body: kernel_body,
            visible: false,
        }
    }

    fn set_visible(mut self) -> Self {
        self.visible = true;
        self
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Module {
    pub directives: ModuleDirectives,
    pub entry: Entry,
}

fn is_followsym(c: u8) -> bool {
    (c >= 'a' as u8 && c <= 'z' as u8) ||
        (c >= 'A' as u8 && c <= 'Z' as u8) ||
        (c >= '0' as u8 && c <= '9' as u8) ||
        (c == '_' as u8) || (c == '$' as u8)
}

fn is_ident_leader(c: u8) -> bool {
    (c == '_' as u8) || (c == '$' as u8) || (c == '%' as u8)
}

named!(identifier(&[u8]) -> String,
    alt!(
        do_parse!(
            leader: map_res!(take_while_m_n!(1, 1, is_alphabetic), str::from_utf8) >>
            rest: map_res!(take_while!(is_followsym), str::from_utf8) >>
            (leader.to_owned() + rest)
        ) |
        do_parse!(
            leader: map_res!(take_while_m_n!(1, 1, is_ident_leader), str::from_utf8) >>
            rest: map_res!(take_while1!(is_followsym), str::from_utf8) >>
            (leader.to_owned() + rest)
        )
    )
);

named!(u8_literal(&[u8]) -> u8,
    map_res!(map_res!(digit, str::from_utf8), |s: &str| s.parse::<u8>())
);

named!(version_directive(&[u8]) -> Version,
    do_parse!(
        tag!(".version") >>
        multispace >>
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
        multispace >>
        // TODO: Extend this to support debug and platform options
        target: target_literal >>
        (target)
    )
);

named!(address_size_directive(&[u8]) -> u8,
    do_parse!(
        tag!(".address_size") >>
        multispace >>
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

named!(variable_type_literal(&[u8]) -> VariableType,
    map_res!(
        map_res!(
            alt!(tag!(".s8") | tag!(".s16") | tag!(".s32") | tag!(".s64") |
                 tag!(".u8") | tag!(".u16") | tag!(".u32") | tag!(".u64") |
                 tag!(".f16") |  tag!(".f32") | tag!(".f64") |
                 tag!(".b8") | tag!(".b16") | tag!(".b32") | tag!(".b64")),
            str::from_utf8
        ),
        |s: &str| s.parse::<VariableType>()
    )
);

named!(param(&[u8]) -> Param,
    do_parse!(
        tag!(".param") >>
        multispace >>
        var_type: variable_type_literal >>
        multispace >>
        name: identifier >>
        (Param { var_type, name })
    )
);

named!(param_list(&[u8]) -> Vec<Param>,
    separated_list!(
        delimited!(opt!(multispace), tag!(","), opt!(multispace)),
        param
    )
);

named!(entry_directive(&[u8]) -> Entry,
    do_parse!(
        tag!(".entry") >>
        multispace >>
        name: identifier >>
        opt!(multispace) >>
        char!('(') >>
        opt!(multispace) >>
        param_list: param_list >>
        opt!(multispace) >>
        char!(')') >>
        opt!(multispace) >>
        char!('{') >>
        opt!(multispace) >>
        kernel_body: map_res!(is_not!("}"), str::from_utf8) >>
        char!('}') >>
        (Entry::new(name, param_list, kernel_body.to_owned()))
    )
);

named!(visible_directive(&[u8]) -> Entry,
    do_parse!(
        tag!(".visible") >>
        multispace >>
        entry: entry_directive >>
        (entry.set_visible())
    )
);

named!(module(&[u8]) -> Module,
    do_parse!(
        directives: module_directives >>
        multispace >>
        entry: visible_directive >>
        (Module { directives, entry })
    )
);

pub fn preprocess(input: &[u8]) -> String {
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

pub fn parse(input: &[u8]) -> Module {
    // TODO: parser error handling, comment stripping error handling, etc.
    let clean = preprocess(input);
    let result = module(clean.as_bytes());
    //match result.err().unwrap() {
    //    nom::Err::Error(nom::simple_errors::Context::Code(c, _)) => println!("{:?}", str::from_utf8(c)),
    //    _ => panic!("crap"),
    //};
    result.unwrap().1
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
    fn is_followsym_test() {
        assert_eq!(is_followsym('a' as u8), true);
        assert_eq!(is_followsym('z' as u8), true);
        assert_eq!(is_followsym('A' as u8), true);
        assert_eq!(is_followsym('Z' as u8), true);
        assert_eq!(is_followsym('0' as u8), true);
        assert_eq!(is_followsym('9' as u8), true);
        assert_eq!(is_followsym('_' as u8), true);
        assert_eq!(is_followsym('$' as u8), true);
        assert_eq!(is_followsym(' ' as u8), false);
    }

    #[test]
    fn is_ident_leader_test() {
        assert_eq!(is_ident_leader('_' as u8), true);
        assert_eq!(is_ident_leader('$' as u8), true);
        assert_eq!(is_ident_leader('%' as u8), true);
        assert_eq!(is_ident_leader(' ' as u8), false);
    }

    #[test]
    fn identifier_test() {
        assert_eq!(identifier(b"a "),
            Ok((
                &b" "[..],
                String::from("a")
            ))
        );
        assert_eq!(identifier(b"abc123 "),
            Ok((
                &b" "[..],
                String::from("abc123")
            ))
        );
        assert_eq!(identifier(b"X "),
            Ok((
                &b" "[..],
                String::from("X")
            ))
        );
        assert_eq!(identifier(b"XabcYZ_$123 "),
            Ok((
                &b" "[..],
                String::from("XabcYZ_$123")
            ))
        );
        assert_eq!(identifier(b"_a "),
            Ok((
                &b" "[..],
                String::from("_a")
            ))
        );
        assert_eq!(identifier(b"$$ "),
            Ok((
                &b" "[..],
                String::from("$$")
            ))
        );
        assert_eq!(identifier(b"%2 "),
            Ok((
                &b" "[..],
                String::from("%2")
            ))
        );
        assert_eq!(identifier(b"$ "),
            Err(nom::Err::Error(error_position!(
                &b"$ "[..],
                nom::ErrorKind::Alt
            )))
        );
        assert_eq!(identifier(b"123 "),
            Err(nom::Err::Error(error_position!(
                &b"123 "[..],
                nom::ErrorKind::Alt
            )))
        );
        assert_eq!(identifier(b"_ "),
            Err(nom::Err::Error(error_position!(
                &b"_ "[..],
                nom::ErrorKind::Alt
            )))
        );
    }

    #[test]
    fn variable_type_literal_test() {
        assert_eq!(variable_type_literal(b".u32"),
            Ok((
                &b""[..],
                VariableType::U32
            ))
        );
        assert_eq!(variable_type_literal(b".u64"),
            Ok((
                &b""[..],
                VariableType::U64
            ))
        );
        assert_eq!(variable_type_literal(b".f32"),
            Ok((
                &b""[..],
                VariableType::F32
            ))
        );
        assert_eq!(variable_type_literal(b".z32"),
            Err(nom::Err::Error(error_position!(
                &b".z32"[..],
                nom::ErrorKind::Alt
            )))
        );
    }

    #[test]
    fn param_test() {
        assert_eq!(param(b".param .u32 abc123,"),
            Ok((
                &b","[..],
                Param {
                    var_type: VariableType::U32,
                    name: String::from("abc123"),
                }
            ))
        );
    }

    #[test]
    fn param_list_test() {
        assert_eq!(param_list(b".param .u32 abc123, .param .f64 xyz789)"),
            Ok((
                &b")"[..],
                vec!(
                    Param {
                        var_type: VariableType::U32,
                        name: String::from("abc123"),
                    },
                    Param {
                        var_type: VariableType::F64,
                        name: String::from("xyz789"),
                    }
                )
            ))
        );
    }

    #[test]
    fn entry_directive_test() {
        assert_eq!(entry_directive(b".entry abc123(.param .u32 abc123) { body body } "),
            Ok((
                &b" "[..],
                Entry {
                    name: String::from("abc123"),
                    param_list: vec!(Param {
                        var_type: VariableType::U32,
                        name: String::from("abc123"),
                    }),
                    kernel_body: String::from("body body "),
                    visible: false,
                }
            ))
        );
    }

    #[test]
    fn visible_directive_test() {
        assert_eq!(visible_directive(b".visible .entry abc123(.param .u32 abc123) { body body } "),
            Ok((
                &b" "[..],
                Entry {
                    name: String::from("abc123"),
                    param_list: vec!(Param {
                        var_type: VariableType::U32,
                        name: String::from("abc123"),
                    }),
                    kernel_body: String::from("body body "),
                    visible: true,
                }
            ))
        );
    }

    #[test]
    fn module_test() {
        assert_eq!(module(include_bytes!("test_ptx/simple_module.ptx")),
            Ok((
                &b"\n"[..],
                Module {
                    directives: ModuleDirectives {
                        version: Version { major: 6, minor: 0 },
                        target: Target::Sm30,
                        address_size: 64u8,
                    },
                    entry: Entry {
                        name: String::from("abc123"),
                        param_list: vec!(Param {
                            var_type: VariableType::U32,
                            name: String::from("abc123"),
                        }),
                        kernel_body: String::from("body body "),
                        visible: true,
                    },
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
