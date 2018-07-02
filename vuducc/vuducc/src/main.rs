extern crate ptx_parser;

fn main() {
    let module = ptx_parser::parse(include_bytes!("../../../saxpy.ptx"));
    println!("{:?}", module);
}
