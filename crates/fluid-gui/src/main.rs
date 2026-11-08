//! Desktop binary entry point.
//!
//! All GUI code lives in the `fluid_gui` library (src/lib.rs) so the same source
//! also builds the Android cdylib. This file only forwards to the library.

fn main() {
    std::process::exit(fluid_gui::run());
}
