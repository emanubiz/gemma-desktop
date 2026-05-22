#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;
use tauri::Manager;

struct BackendProcess(Mutex<Option<Child>>);

fn get_server_pids_from_ps() -> Vec<i32> {
    let output = Command::new("ps")
        .args(["aux"])
        .stdout(Stdio::piped())
        .output();

    let mut pids = Vec::new();
    if let Ok(out) = output {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            if line.contains("mlx_vlm.server") || line.contains("kokoro-server.py") {
                if let Some(pid_str) = line.split_whitespace().nth(1) {
                    if let Ok(pid) = pid_str.parse::<i32>() {
                        pids.push(pid);
                    }
                }
            }
        }
    }
    pids
}

fn cleanup_remaining_servers() {
    // Phase 1: try PID file
    let pid_file = std::path::PathBuf::from("/tmp/gemma-desktop.pids.json");
    let mut killed = Vec::new();
    if let Ok(data) = std::fs::read_to_string(&pid_file) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
            for key in ["gemma", "kokoro"] {
                if let Some(pid_val) = json.get(key).and_then(|v| v.as_i64()) {
                    let pid = pid_val as i32;
                    unsafe {
                        let _ = libc::kill(-pid, libc::SIGTERM);
                    }
                    killed.push(pid);
                }
            }
        }
        let _ = std::fs::remove_file(&pid_file);
    }

    // Phase 2: ps fallback — find any stray server processes
    let stray = get_server_pids_from_ps();
    for pid in &stray {
        if !killed.contains(pid) {
            unsafe {
                let _ = libc::kill(*pid, libc::SIGTERM);
            }
        }
    }

    // Phase 3: wait, then SIGKILL everything still alive
    std::thread::sleep(Duration::from_millis(1000));
    let remaining = get_server_pids_from_ps();
    for pid in &remaining {
        unsafe {
            let _ = libc::kill(*pid, libc::SIGKILL);
        }
    }
}

fn kill_backend_and_servers(backend: &BackendProcess) {
    let mut guard = match backend.0.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let child = match guard.as_mut() {
        Some(c) => c,
        None => return,
    };

    let pid = child.id() as i32;

    // 1) Ask backend to shut down gracefully
    unsafe {
        let _ = libc::kill(pid, libc::SIGTERM);
    }

    // 2) Immediately also try to kill known children via PID file
    cleanup_remaining_servers();

    // 3) Wait up to 5 seconds for backend to exit
    for _ in 0..50 {
        std::thread::sleep(Duration::from_millis(100));
        if child.try_wait().unwrap_or(None).is_some() {
            cleanup_remaining_servers();
            *guard = None;
            return;
        }
    }

    // 4) Backend still alive — force kill it
    let _ = child.kill();
    let _ = child.wait();

    // 5) Final ps sweep
    cleanup_remaining_servers();
    *guard = None;
}

impl Drop for BackendProcess {
    fn drop(&mut self) {
        // Safety net in case on_exit doesn't fire
        kill_backend_and_servers(self);
    }
}

fn main() {
    let child = Command::new("/Users/manu/gemma-env/bin/python")
        .arg("/Users/manu/gemma-desktop/backend.py")
        .spawn()
        .expect("Failed to start Python backend");

    let backend = BackendProcess(Mutex::new(Some(child)));

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(move |app| {
            app.manage(backend);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                if let Some(bp) = app_handle.try_state::<BackendProcess>() {
                    kill_backend_and_servers(&bp);
                }
            }
        });
}
