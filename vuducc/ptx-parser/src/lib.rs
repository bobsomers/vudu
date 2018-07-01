pub fn hello_ptx() {
    println!("Hello, from ptx-parser!");
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
