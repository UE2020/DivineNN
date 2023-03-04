use chess::*;

type EncodedPositions = ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>>;

fn coords(mut sq: Square, flip: bool) -> (usize, usize) {
    if flip {
        sq = unsafe { Square::new((sq.to_index() ^ 0x38) as u8) };
    }

    (sq.get_rank() as usize, sq.get_file() as usize)
}

fn icoords(sq: Square) -> (isize, isize) {
    (sq.get_rank() as isize, sq.get_file() as isize)
}

pub fn encode_positions(boards: &[Board]) -> EncodedPositions {
    let count = boards.len();
    let mut planes = ndarray::Array::<f32, _>::zeros((count, 16, 8, 8));
    for (i, board) in boards.iter().enumerate() {
        let flip = board.side_to_move() == Color::Black;
        let pawns = board.pieces(Piece::Pawn);
        let knights = board.pieces(Piece::Knight);
        let bishops = board.pieces(Piece::Bishop);
        let rooks = board.pieces(Piece::Rook);
        let queens = board.pieces(Piece::Queen);
        let kings = board.pieces(Piece::King);

        let mut white = board.color_combined(Color::White);
        let mut black = board.color_combined(Color::Black);

        if flip {
            std::mem::swap(&mut white, &mut black);
        }

        //////////////////// pawns ////////////////////

        let mut remaining = white & pawns;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();
            let (r, f) = coords(sq, flip);
            planes[[i, 0, r, f]] = 1.0;

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & pawns;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();
            let (r, f) = coords(sq, flip);
            planes[[i, 1, r, f]] = 1.0;

            remaining ^= BitBoard::from_square(sq);
        }

        //////////////////// knights ////////////////////

        let mut remaining = white & knights;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();
            let (r, f) = coords(sq, flip);
            planes[[i, 6, r, f]] = 1.0;

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & knights;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();
            let (r, f) = coords(sq, flip);
            planes[[i, 7, r, f]] = 1.0;

            remaining ^= BitBoard::from_square(sq);
        }

        //////////////////// bishops ////////////////////

        let mut remaining = white & bishops;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();
            let (r, f) = coords(sq, flip);
            planes[[i, 4, r, f]] = 1.0;

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & bishops;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();
            let (r, f) = coords(sq, flip);
            planes[[i, 5, r, f]] = 1.0;

            remaining ^= BitBoard::from_square(sq);
        }

        //////////////////// rooks ////////////////////

        let mut remaining = white & rooks;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();
            let (r, f) = coords(sq, flip);
            planes[[i, 2, r, f]] = 1.0;

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & rooks;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();
            let (r, f) = coords(sq, flip);
            planes[[i, 3, r, f]] = 1.0;

            remaining ^= BitBoard::from_square(sq);
        }

        //////////////////// queens ////////////////////

        let mut remaining = white & queens;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();
            let (r, f) = coords(sq, flip);
            planes[[i, 8, r, f]] = 1.0;

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & queens;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();
            let (r, f) = coords(sq, flip);
            planes[[i, 9, r, f]] = 1.0;

            remaining ^= BitBoard::from_square(sq);
        }

        //////////////////// kings ////////////////////

        let mut remaining = white & kings;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();
            let (r, f) = coords(sq, flip);
            planes[[i, 10, r, f]] = 1.0;

            remaining ^= BitBoard::from_square(sq);
        }

        let mut remaining = black & kings;
        while remaining != BitBoard(0) {
            let sq = remaining.to_square();
            let (r, f) = coords(sq, flip);
            planes[[i, 11, r, f]] = 1.0;

            remaining ^= BitBoard::from_square(sq);
        }

        let mut white_color = Color::White;
        let mut black_color = Color::Black;

        if flip {
            std::mem::swap(&mut white_color, &mut black_color);
        }

        let white_castle_rights = board.castle_rights(white_color);
        let black_castle_rights = board.castle_rights(black_color);

        if white_castle_rights.has_kingside() {
            for x in 0..8 {
                for y in 0..8 {
                    planes[[i, 12, x, y]] = 1.0;
                }
            }
        }

        if black_castle_rights.has_kingside() {
            for x in 0..8 {
                for y in 0..8 {
                    planes[[i, 13, x, y]] = 1.0;
                }
            }
        }

        if white_castle_rights.has_queenside() {
            for x in 0..8 {
                for y in 0..8 {
                    planes[[i, 14, x, y]] = 1.0;
                }
            }
        }

        if black_castle_rights.has_queenside() {
            for x in 0..8 {
                for y in 0..8 {
                    planes[[i, 15, x, y]] = 1.0;
                }
            }
        }
    }

    planes
}

#[allow(unused_assignments)]
pub fn move_to_idx(mut mov: ChessMove, flip: bool) -> (isize, isize, isize) {
    if flip {
        let mov_from = unsafe { Square::new((mov.get_source().to_index() ^ 0x38) as u8) };
        let mov_to = unsafe { Square::new((mov.get_dest().to_index() ^ 0x38) as u8) };
        mov = ChessMove::new(mov_from, mov_to, mov.get_promotion());
    }

    let (from_rank, from_file) = icoords(mov.get_source());
    let (to_rank, to_file) = icoords(mov.get_dest());

    let mut direction_plane = 0;
    let mut distance = 0;
    let mut direction_and_distance_plane = 0;

    if from_rank == to_rank && from_file < to_file {
        direction_plane = 0;
        distance = to_file - from_file;
        direction_and_distance_plane = direction_plane + distance
    } else if from_rank == to_rank && from_file > to_file {
        direction_plane = 8;
        distance = from_file - to_file;
        direction_and_distance_plane = direction_plane + distance
    } else if from_file == to_file && from_rank < to_rank {
        direction_plane = 16;
        distance = to_rank - from_rank;
        direction_and_distance_plane = direction_plane + distance
    } else if from_file == to_file && from_rank > to_rank {
        direction_plane = 24;
        distance = from_rank - to_rank;
        direction_and_distance_plane = direction_plane + distance
    } else if to_file - from_file == to_rank - from_rank && to_file - from_file > 0 {
        direction_plane = 32;
        distance = to_rank - from_rank;
        direction_and_distance_plane = direction_plane + distance
    } else if to_file - from_file == to_rank - from_rank && to_file - from_file < 0 {
        direction_plane = 40;
        distance = from_rank - to_rank;
        direction_and_distance_plane = direction_plane + distance
    } else if to_file - from_file == -(to_rank - from_rank) && to_file - from_file > 0 {
        direction_plane = 48;
        distance = to_file - from_file;
        direction_and_distance_plane = direction_plane + distance
    } else if to_file - from_file == -(to_rank - from_rank) && to_file - from_file < 0 {
        direction_plane = 56;
        distance = from_file - to_file;
        direction_and_distance_plane = direction_plane + distance
    } else if to_file - from_file == 1 && to_rank - from_rank == 2 {
        direction_and_distance_plane = 64
    } else if to_file - from_file == 2 && to_rank - from_rank == 1 {
        direction_and_distance_plane = 65
    } else if to_file - from_file == 2 && to_rank - from_rank == -1 {
        direction_and_distance_plane = 66
    } else if to_file - from_file == 1 && to_rank - from_rank == -2 {
        direction_and_distance_plane = 67
    } else if to_file - from_file == -1 && to_rank - from_rank == 2 {
        direction_and_distance_plane = 68
    } else if to_file - from_file == -2 && to_rank - from_rank == 1 {
        direction_and_distance_plane = 69
    } else if to_file - from_file == -2 && to_rank - from_rank == -1 {
        direction_and_distance_plane = 70
    } else if to_file - from_file == -1 && to_rank - from_rank == -2 {
        direction_and_distance_plane = 71
    }

    (direction_and_distance_plane, from_rank, from_file)
}

type MoveMasks = ndarray::ArrayBase<ndarray::OwnedRepr<i32>, ndarray::Dim<[usize; 4]>>;

pub fn legal_move_masks(boards: &[Board]) -> MoveMasks {
    let count = boards.len();
    let mut masks = ndarray::Array::<i32, _>::zeros((count, 72, 8, 8));

    for (i, board) in boards.iter().enumerate() {
        let flip = board.side_to_move() == Color::Black;

        let movegen = MoveGen::new_legal(&board);
        for mov in movegen {
            let (plane_idx, rank_idx, file_idx) = move_to_idx(mov, flip);
            masks[[i, plane_idx as usize, rank_idx as usize, file_idx as usize]] = 1;
        }
    }

    masks
}

/// Get the policy head probabilities and the value head prediction for a given position.
pub fn get_neural_output(board: Board, network: &tch::CModule) -> (Vec<(ChessMove, f32)>, f32) {
    let position = encode_positions(&[board]);
    let mask = legal_move_masks(&[board]);

    let position: tch::Tensor = tch::Tensor::try_from(position).unwrap();
    let mask: tch::Tensor = tch::Tensor::try_from(mask).unwrap();

    let output = network
        .forward_is(&[
            tch::jit::IValue::Tensor(position),
            tch::jit::IValue::Tensor(mask),
        ])
        .unwrap();

    match output {
        tch::jit::IValue::Tuple(tensors) => {
            let value = match &tensors[0] {
                tch::jit::IValue::Tensor(tensor) => tensor,
                _ => unreachable!(),
            };

            let policy = match &tensors[1] {
                tch::jit::IValue::Tensor(tensor) => tensor,
                _ => unreachable!(),
            };

            let policy = &policy.nan_to_num(0.0, 0.0, 0.0);

            let value: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, _> = value.try_into().unwrap();
            let value = value[[0, 0]];

            let policy: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, _> = policy.try_into().unwrap();
            let flat_policy: Vec<f32> = policy.into_iter().collect::<Vec<_>>();

            let mut move_probabilities = Vec::new();
            let movegen = MoveGen::new_legal(&board);
            for mov in movegen {
                let flip = board.side_to_move() == Color::Black;

                let (plane_idx, rank_idx, file_idx) = move_to_idx(mov, flip);
                let mov_idx = plane_idx * 64 + rank_idx * 8 + file_idx;
                move_probabilities.push((mov, flat_policy[mov_idx as usize]));
            }

            (move_probabilities, value)
        }
        _ => unreachable!(),
    }
}

/// Get the policy head probabilities and the value head prediction for a batch of positions.
pub fn get_neural_output_batched(
    boards: &[Board],
    network: &tch::CModule,
) -> Vec<(Vec<(ChessMove, f32)>, f32)> {
    let positions = encode_positions(boards);
    let masks = legal_move_masks(boards);

    let positions: tch::Tensor = tch::Tensor::try_from(positions).unwrap();
    let masks: tch::Tensor = tch::Tensor::try_from(masks).unwrap();

    let output = network
        .forward_is(&[
            tch::jit::IValue::Tensor(positions),
            tch::jit::IValue::Tensor(masks),
        ])
        .unwrap();

    match output {
        tch::jit::IValue::Tuple(tensors) => {
            let value = match &tensors[0] {
                tch::jit::IValue::Tensor(tensor) => tensor,
                _ => unreachable!(),
            };

            let policy = match &tensors[1] {
                tch::jit::IValue::Tensor(tensor) => tensor,
                _ => unreachable!(),
            };

            let policy = &policy.nan_to_num(0.0, 0.0, 0.0);
            let value: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, _> = value.try_into().unwrap();
            let policy: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, _> = policy.try_into().unwrap();
            let mut outputs = vec![];

            for (i, board) in boards.iter().enumerate() {
                let flip = board.side_to_move() == Color::Black;
                let mut move_probabilities = Vec::new();
                let movegen = MoveGen::new_legal(&board);
                for mov in movegen {
                    let (plane_idx, rank_idx, file_idx) = move_to_idx(mov, flip);
                    let mov_idx = plane_idx * 64 + rank_idx * 8 + file_idx;
                    move_probabilities.push((mov, policy[[i, mov_idx as usize]]));
                }

                outputs.push((move_probabilities, value[[i, 0]]));
            }

            outputs
        }
        _ => unreachable!(),
    }
}
