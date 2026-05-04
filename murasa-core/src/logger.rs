use std::path::Path;

/// Static logger facade wrapping the `log` crate.
pub struct LoggerSetup;

impl LoggerSetup {
    /// Initialize the global logger with an optional file sink.
    pub fn init(name: &str, log_file: Option<&Path>) -> Result<(), log::SetLoggerError> {
        let mut builder = env_logger::Builder::from_default_env();
        builder.format(|buf, record| {
            use std::io::Write;
            writeln!(
                buf,
                "{} | {:<8} | {}",
                buf.timestamp_seconds(),
                record.level(),
                record.args()
            )
        });
        builder.target(env_logger::Target::Stdout);
        if let Some(path) = log_file {
            // In a real implementation we would add a file appender here.
            let _ = path;
        }
        builder.try_init()
    }

    /// Get a logger instance for the given target.
    pub fn get_logger(name: &str) -> &'static dyn log::Log {
        log::logger()
    }
}

/// Emit a section header.
pub fn log_section(title: &str, width: usize) {
    let line = "=".repeat(width);
    let centered = format!("{:^width$}", title, width = width);
    log::info!("\n{}\n{}\n{}", line, centered, line);
}

/// Emit a subsection header.
pub fn log_subsection(title: &str, width: usize) {
    let line = "-".repeat(width);
    log::info!("\n{}\n{}\n{}", line, title, line);
}

/// Log a success message.
pub fn log_success(message: &str) {
    log::info!("{}", message);
}

/// Log an error message.
pub fn log_error(message: &str) {
    log::error!("{}", message);
}

/// Log a warning message.
pub fn log_warning(message: &str) {
    log::warn!("{}", message);
}

/// Log progress.
pub fn log_progress(current: usize, total: usize, prefix: &str) {
    let pct = (current as f64 / total as f64) * 100.0;
    log::info!("{} [{}/{}] {:.1}%", prefix, current, total, pct);
}
