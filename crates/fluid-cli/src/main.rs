//! fluid: real-time frame generation and upscaling for any game, video or screen.
//!
//! This binary is a thin shell around fluid-engine: it parses arguments, loads
//! configuration, opens a capture source and a present sink, then runs the
//! pipeline until the source ends or the user closes the window.
//!
//! Ctrl+C / Ctrl+Break are handled gracefully (see ctrl): the handler asks the
//! pipeline to stop at the next frame boundary, so the final report is still
//! written instead of the process dying with 0xC000013A.

mod cli;
mod commands;
mod ctrl;
mod dpi;

use clap::Parser;

fn main() {
    // Before anything queries a monitor, a window or a capture size.
    dpi::enable();
    let cli = cli::Cli::parse();
    // Install before doing any work so even an early Ctrl+C stays graceful.
    let cancel = ctrl::install();
    if let Err(err) = commands::dispatch(&cli, &cancel) {
        eprintln!("fluid: error: {err}");
        std::process::exit(1);
    }
}
