// Known failed positions:
// 4b3/1k6/8/1PB1K3/2pNp1p1/5p2/1PP4P/8 w - - 0 55
// 8/5p1p/k3r3/p1Q5/3P4/8/P4nP1/1K6 w - - 1 52

use super::*;

use std::cell::RefCell;
use std::process::Child;
use std::rc::Rc;

pub fn calculate_uct(edge: &Edge, n_p: f32, root: bool) -> f32 {
    let q = edge.get_q();
    let n_c = edge.get_n();
    let p = edge.p;

    /*let init = 1.745;
    let factor = 3.894;

    let c = if root {
        init + factor * ((n_c + 38739.0) / 38739.00).ln()
    } else {
        3.1
    };*/
    let uct = q + p * 1.5 * n_p.sqrt() / (1.0 + n_c);

    uct
}

#[allow(unused)]
pub fn calculate_uct_no_cpuct(edge: &Edge, n_p: f32) -> f32 {
    let q = edge.get_q();
    let n_c = edge.get_n();
    let p = edge.p;

    let uct = q + p * 0.0 * n_p.sqrt() / (1.0 + n_c);

    uct
}

#[derive(Clone, Debug)]
pub struct Node {
    pub n: f32,
    sum_q: f32,
    edges: Vec<Rc<RefCell<Edge>>>,
}

impl Node {
    pub fn new(
        new_q: f32,
        probabilities: &mut [(ChessMove, f32)],
    ) -> Self {
        Self {
            n: 1.0,
            sum_q: new_q,
            edges: {
                let mut probabilities = probabilities.to_vec();
                probabilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let mut total = 0.0;
                for probability in probabilities.iter() {
                    total += probability.1;
                }

                let edges = probabilities
                    .into_iter()
                    .map(|p| Rc::new(RefCell::new(Edge::new(p.0, p.1 / total))))
                    .collect::<Vec<_>>();

                edges
            },
        }
    }

    pub fn get_q(&self) -> f32 {
        self.sum_q / self.n
    }

    pub fn uct_select(&self, root: bool) -> Option<Rc<RefCell<Edge>>> {
        let mut max_uct = -1000.0;
        let mut max_edge = None;

        for edge in self.edges.iter() {
            let uct = calculate_uct(&edge.borrow(), self.n, root);
            if max_uct < uct {
                max_uct = uct;
                max_edge = Some(edge.clone());
            }
        }

        max_edge
    }

    pub fn max_n_select(&self, game: &Game, detect_draw: bool) -> Option<Rc<RefCell<Edge>>> {
        let mut max_n = -1.0;
        let mut max_edge = None;
        let score = ((self.get_q() - 0.5) * 15.0 * 100.0) as i32;

        'outer: for edge in self.edges.iter() {
            let unlocked_edge = edge.borrow();
            if score >= 0 && detect_draw {
                // try the move
                let mut game = game.clone();
                game.make_move(unlocked_edge.mov);
                if game.can_declare_draw() {
                    println!("Draw possible");
                    continue;
                } else {
                    let movegen = MoveGen::new_legal(&game.current_position());
                    for mov in movegen {
                        let mut game = game.clone();
                        game.make_move(mov);
                        if game.can_declare_draw() {
                            continue 'outer;
                        }
                    }
                }
            }

           // let value = calculate_uct_no_cpuct(&edge.borrow(), self.n);
            //let value = unlocked_edge.get_q();
            let value = unlocked_edge.get_n();
            if max_n < value {
                max_n = value;
                max_edge = Some(edge.clone());
            }
        }

        if max_edge.is_none() && detect_draw {
            max_edge = self.max_n_select(game, false);
        }

        max_edge
    }

    pub fn is_terminal(&self) -> bool {
        self.edges.len() == 0
    }
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub mov: ChessMove,
    p: f32,
    pub child: Option<Rc<RefCell<Node>>>,
    virtual_losses: f32,
}

impl Edge {
    pub fn new(mov: ChessMove, probability: f32) -> Self {
        Self {
            mov,
            p: probability,
            child: None,
            virtual_losses: 0.0,
        }
    }

    pub fn has_child(&self) -> bool {
        self.child.is_some()
    }

    pub fn get_n(&self) -> f32 {
        if let Some(child) = &self.child {
            child.borrow().n + self.virtual_losses
        } else {
            self.virtual_losses
        }
    }

    pub fn get_q(&self) -> f32 {
        if let Some(child) = &self.child {
            let child = child.borrow();
            1.0 - ((child.sum_q + self.virtual_losses) / (child.n + self.virtual_losses))
        } else {
            0.0
        }
    }

    pub fn expand(
        &mut self,
        new_q: f32,
        move_probabilities: &mut [(ChessMove, f32)],
    ) -> bool {
        if self.child.is_none() {
            self.child = Some(Rc::new(RefCell::new(Node::new(
                new_q,
                move_probabilities,
            ))));

            true
        } else {
            false
        }
    }

    pub fn add_virtual_loss(&mut self) {
        self.virtual_losses += 1.0;
    }

    pub fn clear_virtual_loss(&mut self) {
        self.virtual_losses = 0.0;
    }
}

#[derive(Clone, Debug)]
pub struct Root {
    root_node: Rc<RefCell<Node>>,
    pub depth: usize,
    pub same_paths: usize,
}

impl Root {
    pub fn new(board: Board, network: &tch::CModule) -> Self {
        let (mut move_probabilities, value) = get_neural_output(board, network);
        let q = value / 2.0 + 0.5;
        let node = Node::new(q, &mut move_probabilities);

        Self {
            root_node: Rc::new(RefCell::new(node)),
            same_paths: 0,
            depth: 0,
        }
    }

    pub fn root_node(&self) -> Rc<RefCell<Node>> {
        self.root_node.clone()
    }

    pub fn select_task(
        root_node: Rc<RefCell<Node>>,
        board: &mut Board,
        node_path: &mut Vec<Rc<RefCell<Node>>>,
        edge_path: &mut Vec<Option<Rc<RefCell<Edge>>>>,
    ) {
        let mut c_node = root_node.clone();
        let mut is_root = true;
        loop {
            node_path.push(c_node.clone());
            let c_edge = c_node.borrow().uct_select(is_root);
            edge_path.push(c_edge.clone());

            if c_edge.is_none() {
                assert!(c_node.borrow().is_terminal());
                break;
            }

            let c_edge = c_edge.unwrap();
            let mut c_edge = c_edge.borrow_mut();

            c_edge.add_virtual_loss();
            *board = board.make_move_new(c_edge.mov);

            if !c_edge.has_child() {
                break;
            }

            c_node = c_edge.child.as_ref().unwrap().clone();
            is_root = false;
        }
    }

    pub fn parallel_rollouts(
        &mut self,
        board: Board,
        network: &tch::CModule,
        count: usize,

        #[allow(unused)]
        child: Option<&mut Child>,
    ) {
        let mut results = vec![];
        for _ in 0..count {
            let root_node = self.root_node().clone();
            let mut job = Job::new(board);
            Self::select_task(
                root_node,
                &mut job.board,
                &mut job.node_path,
                &mut job.edge_path,
            );
            results.push(job);
        }

        let mut boards = vec![];
        for result in results.iter() {
            boards.push(result.board);
        }

        let mut output = get_neural_output_batched(&boards, &network);

        #[cfg(feature = "use-external-eval")]
        let child = child.unwrap();

        for (job, output) in results.iter().zip(output.iter_mut()) {
            let edges = job.edge_path.len();
            let edge = &job.edge_path[edges - 1];
            let board = job.board;
            let mut new_q;

            if let Some(edge) = edge {
                // get the value
                let value = match board.status() {
                    BoardStatus::Checkmate => {
                        let mut winner = if board.side_to_move() == Color::White {
                            -1.0
                        } else {
                            1.0
                        };

                        if board.side_to_move() != Color::White {
                            winner *= -1.0;
                        }

                        winner
                    }
                    BoardStatus::Stalemate => 0.0,
                    #[cfg(feature = "use-external-eval")]
                    BoardStatus::Ongoing => {
                        use std::io::{Read, Write};
                        use vampirc_uci::UciInfoAttribute;

                        let stdin = child.stdin.as_mut().unwrap();
                        let stdout = child.stdout.as_mut().unwrap();
                        stdin
                            .write_all(format!("position fen {}\n", board.to_string()).as_bytes())
                            .expect("Failed to write to stdin");
                        stdin
                            .write_all("go movetime 25\n".as_bytes())
                            .expect("Failed to write to stdin");

                        let mut last_value = 0.0;
                        loop {
                            let mut bytes = vec![];
                            loop {
                                // read a char
                                let mut output = [0];
                                stdout
                                    .read_exact(&mut output)
                                    .expect("Failed to read output");
                                if output[0] as char == '\n' {
                                    break;
                                }
                                bytes.push(output[0]);
                            }
                            let output = String::from_utf8_lossy(&bytes);
                            let msg = parse_one(&output);
                            match msg {
                                UciMessage::Info(attrs) => {
                                    for attr in attrs {
                                        match attr {
                                            UciInfoAttribute::Score { cp, mate, .. } => {
                                                if let Some(cp) = cp {
                                                    last_value = 2.0
                                                        * (1.0
                                                            / (1.0
                                                                + 10.0f32.powf(
                                                                    -(cp as f32 / 100.0) / 4.0,
                                                                )))
                                                        - 1.0;
                                                } else if let Some(mate) = mate {
                                                    if mate > 0 {
                                                        last_value =
                                                            1.0 - (mate.abs() as f32 * 0.01);
                                                    } else {
                                                        last_value =
                                                            -1.0 + (mate.abs() as f32 * 0.01);
                                                    }
                                                    stdin
                                                        .write_all("stop\n".as_bytes())
                                                        .expect("Failed to write to stdin");
                                                    break;
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                UciMessage::BestMove { .. } => break,
                                _ => {}
                            }
                        }

                        last_value
                    }

                    #[cfg(not(feature = "use-external-eval"))]
                    BoardStatus::Ongoing => output.1,
                };

                new_q = value / 2.0 + 0.5;
                let is_unexpanded = edge.borrow_mut().expand(new_q, &mut output.0);

                if !is_unexpanded {
                    self.same_paths += 1;
                }
                new_q = 1. - new_q
            } else {
                let mut winner = match board.status() {
                    BoardStatus::Checkmate => {
                        if board.side_to_move() == Color::White {
                            -1
                        } else {
                            1
                        }
                    }
                    BoardStatus::Stalemate => 0,
                    BoardStatus::Ongoing => unreachable!(),
                };

                if board.side_to_move() != Color::White {
                    winner *= -1;
                }

                new_q = winner as f32 / 2. + 0.5;
            }

            self.depth = self.depth.max(job.node_path.len());

            let last_node_idx = job.node_path.len() - 1;
            for i in (0..=last_node_idx).rev() {
                let node = &job.node_path[i];
                let mut node = node.borrow_mut();
                node.n += 1.0;
                if (last_node_idx - i) % 2 == 0 {
                    node.sum_q += new_q;
                } else {
                    node.sum_q += 1.0 - new_q;
                }
            }

            for edge in job.edge_path.iter() {
                if let Some(edge) = edge {
                    edge.borrow_mut().clear_virtual_loss();
                }
            }
        }
    }
}

pub struct Job {
    board: Board,
    node_path: Vec<Rc<RefCell<Node>>>,
    edge_path: Vec<Option<Rc<RefCell<Edge>>>>,
}

impl Job {
    pub fn new(board: Board) -> Self {
        Self {
            board,
            node_path: vec![],
            edge_path: vec![],
        }
    }
}
