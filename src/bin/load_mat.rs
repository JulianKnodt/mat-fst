use std::{
  fs::File,
  io::{BufRead, BufReader},
};

fn main() {
  let file_name = "weights.txt";
  let file = File::open(file_name).unwrap();
  let buf = BufReader::new(file);
  for line in buf.lines() {
    let line = line.unwrap();
    let mut parts = line.split_whitespace();
    let x = parts.next().unwrap();
    let x = x.parse::<u16>().unwrap();
    let y = parts.next().unwrap();
    let y = y.parse::<u16>().unwrap();
    let v = parts.next().unwrap();
    let v = v.parse::<f32>().unwrap();
  }
}
