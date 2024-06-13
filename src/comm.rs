use tokio::sync::mpsc;
use tokio::task;
use tokio::time::{sleep, Duration};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // Create a shared state or pool (just a dummy Arc for illustration)
    let shared_state = Arc::new(());

    // Create channels for communication between threads
    let (cmd_tx, cmd_rx) = mpsc::channel(32);
    let (write_tx, write_rx) = mpsc::channel(32);

    // Spawn the command thread
    let cmd_state = shared_state.clone();
    task::spawn(async move {
        command_thread(cmd_state, cmd_tx).await;
    });

    // Spawn the writer thread
    let write_state = shared_state.clone();
    task::spawn(async move {
        writer_thread(write_state, write_rx).await;
    });

    // Spawn the reader thread
    let read_state = shared_state;
    task::spawn(async move {
        reader_thread(read_state, cmd_rx, write_tx).await;
    });

    // Keep the main thread alive for a while to let the threads do some work
    sleep(Duration::from_secs(10)).await;
}

async fn command_thread(state: Arc<()>, cmd_tx: mpsc::Sender<&'static str>) {
    let commands = vec!["Command1", "Command2", "Command3"];
    for cmd in commands {
        // Simulate providing a command
        if let Err(_) = cmd_tx.send(cmd).await {
            println!("Failed to send command: {}", cmd);
        }
        sleep(Duration::from_secs(1)).await;
    }
}

async fn writer_thread(state: Arc<()>, mut write_rx: mpsc::Receiver<&'static str>) {
    while let Some(data) = write_rx.recv().await {
        // Simulate writing data
        println!("Writing data: {}", data);
        sleep(Duration::from_secs(1)).await;
    }
}

async fn reader_thread(
    state: Arc<()>,
    mut cmd_rx: mpsc::Receiver<&'static str>,
    write_tx: mpsc::Sender<&'static str>,
) {
    while let Some(cmd) = cmd_rx.recv().await {
        // Simulate reading data and passing it to the writer
        println!("Reading command: {}", cmd);
        if let Err(_) = write_tx.send(cmd).await {
            println!("Failed to send data to writer: {}", cmd);
        }
        sleep(Duration::from_secs(1)).await;
    }
}