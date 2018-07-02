#[macro_use]
extern crate nom;

use nom::{is_digit, is_space};
use std::str;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Version {
    major: u8,
    minor: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParseTargetError {
    UnknownTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    Sm30,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParseAddressSizeError {
    UnknownAddressSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AddressSize {
    Bit32,
    Bit64,
}

#[derive(Debug, PartialEq, Eq)]
struct Module {
    version: Version,
    target: Target,
    address_size: AddressSize,
}

fn u8_from_bytes(input: &[u8]) -> Result<u8, std::num::ParseIntError> {
    u8::from_str_radix(str::from_utf8(input).unwrap(), 10)
}

fn target_from_bytes(input: &[u8]) -> Result<Target, ParseTargetError> {
    let arch = str::from_utf8(input).unwrap();
    match arch {
        "sm_30" => Ok(Target::Sm30),
        _ => Err(ParseTargetError::UnknownTarget),
    }
}

fn address_size_from_bytes(input: &[u8]) -> Result<AddressSize, ParseAddressSizeError> {
    let size = str::from_utf8(input).unwrap();
    match size {
        "32" => Ok(AddressSize::Bit32),
        "64" => Ok(AddressSize::Bit64),
        _ => Err(ParseAddressSizeError::UnknownAddressSize),
    }
}

named!(line_comment,
    delimited!(
        tag!("//"),
        take_until!("\n"),
        tag!("\n")
    )
);

named!(version_directive<&[u8], Version>,
    do_parse!(
        tag!(".version") >>
        take_while!(is_space) >>
        major: map_res!(take_while!(is_digit), u8_from_bytes) >>
        char!('.') >>
        minor: map_res!(take_while!(is_digit), u8_from_bytes) >>
        tag!("\n") >>
        (Version { major, minor: minor })
    )
);

named!(target_directive<&[u8], Target>,
    do_parse!(
        tag!(".target") >>
        take_while!(is_space) >>
        // TODO: Extend this to support multiple architectures and platform
        //       options
        target: map_res!(tag!("sm_30"), target_from_bytes) >>
        tag!("\n") >>
        (target)
    )
);

named!(address_size_directive<&[u8], AddressSize>,
    do_parse!(
        tag!(".address_size") >>
        take_while!(is_space) >>
        address_size: map_res!(alt!(tag!("32") | tag!("64")), address_size_from_bytes) >>
        tag!("\n") >>
        (address_size)
    )
);

#[cfg(test)]
mod tests {
    use super::*;
    use nom::Err::Incomplete;
    use nom::Needed;

    #[test]
    fn line_comment_test() {
        assert_eq!(line_comment(b"// foo bar\nbaz"),
                   Ok((&b"baz"[..], &b" foo bar"[..])));
        assert_eq!(line_comment(b"// foo bar"),
                   Err(Incomplete(Needed::Size(1))));
    }

    #[test]
    fn version_directive_test() {
        assert_eq!(version_directive(b".version 6.0\nfoo"),
                   Ok((&b"foo"[..], Version { major: 6, minor: 0 })));
        assert_eq!(version_directive(b".version 12.42\n"),
                   Ok((&b""[..], Version { major: 12, minor: 42 })));
        assert_eq!(version_directive(b".version 6\n"),
                   Err(nom::Err::Error(error_position!(&b"\n"[..],
                                                       nom::ErrorKind::Char))));
        assert_eq!(version_directive(b".version 6.\n"),
                   Err(nom::Err::Error(error_position!(&b"\n"[..],
                                                       nom::ErrorKind::MapRes))));
        assert_eq!(version_directive(b".version a.b\n"),
                   Err(nom::Err::Error(error_position!(&b"a.b\n"[..],
                                                       nom::ErrorKind::MapRes))));
        assert_eq!(version_directive(b".version 6.b\n"),
                   Err(nom::Err::Error(error_position!(&b"b\n"[..],
                                                       nom::ErrorKind::MapRes))));
    }

    #[test]
    fn target_directive_test() {
        assert_eq!(target_directive(b".target sm_30\nfoo"),
                   Ok((&b"foo"[..], Target::Sm30)));
        assert_eq!(target_directive(b".target sm_60\n"),
                   Err(nom::Err::Error(error_position!(&b"sm_60\n"[..],
                                                       nom::ErrorKind::Tag))));
                                                    
    }

    #[test]
    fn address_size_directive_test() {
        assert_eq!(address_size_directive(b".address_size 32\nfoo"),
                   Ok((&b"foo"[..], AddressSize::Bit32)));
        assert_eq!(address_size_directive(b".address_size 64\n"),
                   Ok((&b""[..], AddressSize::Bit64)));
        assert_eq!(address_size_directive(b".address_size foo\n"),
                   Err(nom::Err::Error(error_position!(&b"foo\n"[..],
                                                       nom::ErrorKind::Alt))));
    }
}
