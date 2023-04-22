use chess::*;
use std::{
    io::{self, BufRead},
    str::FromStr,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
    sync::{mpsc, Arc},
    thread
};

use vampirc_uci::{UciPiece, UciTimeControl, parse_one, UciMessage};

mod config;
mod mcts;
pub mod encoding;

use config::*;
pub use encoding::*;

fn main() {
    eprintln!("Divine 0.1 compiled on rustc 1.67.0-nightly (09508489e 2022-11-04)");
    eprintln!(
        "Current libtorch intra-op threads: {}",
        tch::get_num_threads()
    );

    let mut board = Game::new();
    let mut model = tch::CModule::load(MODEL).expect("model is in path");
    model.set_eval();
    let model = Arc::new(model);

    eprintln!("Using network: '{}'\n", MODEL);

    // worker
    let (tx, rx) = mpsc::channel();
    let should_stop = Arc::new(AtomicBool::new(false));

    {
        let model = model.clone();
        let should_stop = should_stop.clone();
        thread::spawn(move || {
            loop {
                #[allow(unused)]
                use std::process::{Command, Stdio};

                #[cfg(feature = "use-external-eval")]
                let mut child = Command::new(ENGINE)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .spawn()
                    .expect("failed to execute child");

                let recv: (Game, Option<UciTimeControl>) = rx.recv().unwrap();
                should_stop.store(false, Ordering::Relaxed);
                let board = recv.0;
                let time_control = recv.1;
                let mut root = mcts::Root::new(board.current_position(), &model);
                let now = Instant::now();
                let target = match time_control {
                    Some(time) => match time {
                        UciTimeControl::MoveTime(duration) => duration.to_std().unwrap(),
                        UciTimeControl::TimeLeft {
                            white_time,
                            black_time,
                            white_increment,
                            ..
                        } => {
                            let time_left = match board.side_to_move() {
                                Color::White => {
                                    white_time.unwrap().to_std().unwrap()
                                }
                                Color::Black => {
                                    black_time.unwrap().to_std().unwrap()
                                }
                            };

                            /*let moves = board.actions().len() as f32 / 2.0;
                            let moves_left = if moves < 60.0 {
                                ((-2.0 / 3.0) * moves + 50.0) as u32
                            } else {
                                ((1.0 / 10.0) * (moves - 60.0) + 10.0) as u32
                            };*/
                            (time_left / 40).min(Duration::from_secs(60))
                                + white_increment.unwrap_or(vampirc_uci::Duration::milliseconds(0)).to_std().unwrap()
                        }
                        _ => Duration::from_millis(60000),
                    },
                    None => Duration::from_millis(60000),
                };

                let mut rollouts = 0;
                tch::no_grad(|| loop {
                    // make sure that a sensical move is chosen when time is low
                    if now.elapsed() >= target {
                        break;
                    }

                    root.parallel_rollouts(board.current_position(), &model, 8, {
                        cfg_if::cfg_if! {
                            if #[cfg(feature = "use-external-eval")] {
                                Some(&mut child)
                            } else {
                                None
                            }
                        }
                    });
                    if should_stop.load(Ordering::Relaxed) {
                        should_stop.store(false, Ordering::Relaxed);
                        break;
                    }
                    let edge = root.root_node();
                    let edge = edge.borrow();
                    rollouts += 8;
                    let q = edge.get_q() * 2.0 - 1.0;
                    let score = -(q.signum() * (1.0 - q.abs()).ln() / (1.2f32).ln()) * 100.0 / 2.0;

                    let pv = {
                        let mut pv = vec![];
                        let mut current_node = root.root_node();
                        let mut pv_game = board.clone();
                        loop {
                            let node = current_node.borrow();
                            let edge = node.max_n_select(&pv_game, true);
                            if let Some(edge) = edge {
                                let edge = edge.borrow();
                                let best_move = edge.mov;
                                let best_move = if let Some(_) = best_move.get_promotion() {
                                    ChessMove::new(
                                        best_move.get_source(),
                                        best_move.get_dest(),
                                        Some(Piece::Queen),
                                    )
                                } else {
                                    best_move
                                };
                                pv.push(best_move);
                                pv_game.make_move(best_move);
                                drop(node);
                                if let Some(child) = &edge.child {
                                    current_node = child.clone();
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }

                        pv.iter()
                            .map(|mov| format!("{}", mov))
                            .collect::<Vec<_>>()
                            .join(" ")
                    };

                    println!(
                        "info currmove {} depth {} score cp {} nodes {} nps {} time {} pv {}",
                        edge.max_n_select(&board, true).unwrap().borrow().mov,
                        root.depth,
                        score as i32,
                        rollouts,
                        rollouts as u32 / now.elapsed().as_secs().max(1) as u32,
                        now.elapsed().as_millis(),
                        pv
                    );
                });
                let mut pv = vec![];
                let mut current_node = root.root_node();
                let mut pv_game = board.clone();
                loop {
                    let node = current_node.borrow();
                    let edge = node.max_n_select(&pv_game, true);
                    if let Some(edge) = edge {
                        let edge = edge.borrow();
                        let best_move = edge.mov;
                        let best_move = if let Some(_) = best_move.get_promotion() {
                            ChessMove::new(
                                best_move.get_source(),
                                best_move.get_dest(),
                                Some(Piece::Queen),
                            )
                        } else {
                            best_move
                        };
                        pv.push(best_move);
                        pv_game.make_move(best_move);
                        drop(node);
                        if let Some(child) = &edge.child {
                            current_node = child.clone();
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                let best_move = pv[0];
                let pv = pv
                    .iter()
                    .map(|mov| format!("{}", mov))
                    .collect::<Vec<_>>()
                    .join(" ");
                println!(
                    "info currmove {} depth {} nodes {} time {} pv {}",
                    best_move,
                    root.depth,
                    rollouts,
                    now.elapsed().as_millis(),
                    pv
                );

                println!("bestmove {}", best_move);

                #[cfg(feature = "use-external-eval")]
                {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        });
    }

    for line in io::stdin().lock().lines() {
        let msg: UciMessage = parse_one(&line.unwrap());
        match msg {
            UciMessage::Uci => {
                println!("id name DivineNN");
                println!("uciok")
            }
            UciMessage::Position {
                startpos,
                moves,
                fen,
            } => {
                if startpos {
                    board = Game::new();
                } else if let Some(fen) = fen {
                    board = Game::from_str(&fen.0).unwrap();
                }

                for mov in moves {
                    let from = mov.from;
                    let to = mov.to;

                    let from = Square::make_square(
                        match from.rank {
                            1 => Rank::First,
                            2 => Rank::Second,
                            3 => Rank::Third,
                            4 => Rank::Fourth,
                            5 => Rank::Fifth,
                            6 => Rank::Sixth,
                            7 => Rank::Seventh,
                            8 => Rank::Eighth,
                            _ => unreachable!(),
                        },
                        match from.file {
                            'a' => File::A,
                            'b' => File::B,
                            'c' => File::C,
                            'd' => File::D,
                            'e' => File::E,
                            'f' => File::F,
                            'g' => File::G,
                            'h' => File::H,
                            _ => unreachable!(),
                        },
                    );

                    let to = Square::make_square(
                        match to.rank {
                            1 => Rank::First,
                            2 => Rank::Second,
                            3 => Rank::Third,
                            4 => Rank::Fourth,
                            5 => Rank::Fifth,
                            6 => Rank::Sixth,
                            7 => Rank::Seventh,
                            8 => Rank::Eighth,
                            _ => unreachable!(),
                        },
                        match to.file {
                            'a' => File::A,
                            'b' => File::B,
                            'c' => File::C,
                            'd' => File::D,
                            'e' => File::E,
                            'f' => File::F,
                            'g' => File::G,
                            'h' => File::H,
                            _ => unreachable!(),
                        },
                    );
                    let mov = ChessMove::new(
                        from,
                        to,
                        mov.promotion.map(|piece| match piece {
                            UciPiece::Pawn => Piece::Pawn,
                            UciPiece::Knight => Piece::Knight,
                            UciPiece::Bishop => Piece::Bishop,
                            UciPiece::Rook => Piece::Rook,
                            UciPiece::Queen => Piece::Queen,
                            UciPiece::King => Piece::King,
                        }),
                    );
                    board.make_move(mov);
                }
            }
            UciMessage::Go { time_control, .. } => {
                tx.send((board.clone(), time_control)).unwrap();
            }
            UciMessage::IsReady => println!("readyok"),
            UciMessage::Quit => break,
            UciMessage::Stop => {
                should_stop.store(true, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}
