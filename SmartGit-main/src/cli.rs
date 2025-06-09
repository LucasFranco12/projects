use owo_colors::OwoColorize;
use indicatif::{ProgressBar, ProgressStyle};

pub fn success(msg: &str) {
    println!("{} {}", "✅".green(), msg.green());
}

pub fn warning(msg: &str) {
    println!("{} {}", "⚠️".yellow(), msg.yellow());
}

pub fn error(msg: &str) {
    eprintln!("{} {}", "❌".red(), msg.red());
}

pub fn info(msg: &str) {
    println!("{} {}", "ℹ️".blue(), msg.blue());
}

pub fn create_progress_bar(len: u64, msg: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(msg.to_string());
    pb
}

pub fn create_spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message(msg.to_string());
    pb
} 