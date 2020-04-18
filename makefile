PHONY:

bench-max:
	RUSTFLAGS='-C target-cpu=native -C opt-level=3 -C codegen-units=1' cargo bench --all-features
