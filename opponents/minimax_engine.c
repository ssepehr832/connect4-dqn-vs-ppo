#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <pthread.h>

#define ROWS 6
#define COLS 7
#define WIN_LENGTH 4

/* ---- helpers ---- */

static int get_drop_row(const int board[ROWS][COLS], int col) {
    for (int r = ROWS - 1; r >= 0; r--) {
        if (board[r][col] == 0) return r;
    }
    return -1;
}

static int is_full(const int board[ROWS][COLS]) {
    for (int c = 0; c < COLS; c++) {
        if (board[0][c] == 0) return 0;
    }
    return 1;
}

static int check_direction(const int board[ROWS][COLS], int r, int c,
                           int dr, int dc, int player) {
    int count = 0;
    for (int i = 0; i < WIN_LENGTH; i++) {
        int nr = r + dr * i;
        int nc = c + dc * i;
        if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) return 0;
        if (board[nr][nc] != player) return 0;
        count++;
    }
    return count == WIN_LENGTH;
}

static int check_winner(const int board[ROWS][COLS]) {
    static const int dirs[4][2] = {{0,1},{1,0},{1,1},{1,-1}};
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            if (board[r][c] == 0) continue;
            int p = board[r][c];
            for (int d = 0; d < 4; d++) {
                if (check_direction(board, r, c, dirs[d][0], dirs[d][1], p))
                    return p;
            }
        }
    }
    return 0;
}

/* ---- heuristic evaluation ---- */

static int score_window(int me_count, int opp_count, int empty_count) {
    if (me_count == 4) return 10000;
    if (me_count == 3 && empty_count == 1) return 5;
    if (me_count == 2 && empty_count == 2) return 2;
    if (opp_count == 3 && empty_count == 1) return -4;
    return 0;
}

static int evaluate(const int board[ROWS][COLS], int me) {
    int opp = 3 - me;
    int score = 0;

    /* Center column preference */
    int center = COLS / 2;
    for (int r = 0; r < ROWS; r++) {
        if (board[r][center] == me) score += 3;
    }

    /* Score all windows */
    static const int dirs[4][2] = {{0,1},{1,0},{1,1},{1,-1}};
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            for (int d = 0; d < 4; d++) {
                int me_count = 0, opp_count = 0, empty_count = 0;
                int valid = 1;
                for (int i = 0; i < WIN_LENGTH; i++) {
                    int nr = r + dirs[d][0] * i;
                    int nc = c + dirs[d][1] * i;
                    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) {
                        valid = 0;
                        break;
                    }
                    if (board[nr][nc] == me) me_count++;
                    else if (board[nr][nc] == opp) opp_count++;
                    else empty_count++;
                }
                if (valid)
                    score += score_window(me_count, opp_count, empty_count);
            }
        }
    }
    return score;
}

/* ---- minimax with alpha-beta ---- */

static int minimax(int board[ROWS][COLS], int depth, int alpha, int beta,
                   int is_maximizing, int me) {
    int opp = 3 - me;

    int winner = check_winner(board);
    if (winner == me)  return  10000 + depth;
    if (winner == opp) return -10000 - depth;
    if (is_full(board)) return 0;
    if (depth == 0) return evaluate(board, me);

    if (is_maximizing) {
        int max_eval = INT_MIN;
        for (int c = 0; c < COLS; c++) {
            int r = get_drop_row(board, c);
            if (r < 0) continue;
            board[r][c] = me;
            int val = minimax(board, depth - 1, alpha, beta, 0, me);
            board[r][c] = 0;
            if (val > max_eval) max_eval = val;
            if (val > alpha) alpha = val;
            if (beta <= alpha) break;
        }
        return max_eval;
    } else {
        int min_eval = INT_MAX;
        for (int c = 0; c < COLS; c++) {
            int r = get_drop_row(board, c);
            if (r < 0) continue;
            board[r][c] = opp;
            int val = minimax(board, depth - 1, alpha, beta, 1, me);
            board[r][c] = 0;
            if (val < min_eval) min_eval = val;
            if (val < beta) beta = val;
            if (beta <= alpha) break;
        }
        return min_eval;
    }
}

/* ---- public API ----
 *
 * board_flat: row-major int array of size 42 (6x7)
 * depth:      search depth
 * player:     current player (1 or 2)
 *
 * Returns: best column (0-6)
 */
int minimax_best_action(int *board_flat, int depth, int player) {
    /* Copy flat array into 2D */
    int board[ROWS][COLS];
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            board[r][c] = board_flat[r * COLS + c];

    int best_score = INT_MIN;
    int best_col = -1;
    /* Track ties for random tie-breaking: store up to COLS best actions */
    int best_actions[COLS];
    int num_best = 0;

    for (int c = 0; c < COLS; c++) {
        int r = get_drop_row(board, c);
        if (r < 0) continue;
        board[r][c] = player;
        int score = minimax(board, depth - 1, INT_MIN, INT_MAX, 0, player);
        board[r][c] = 0;

        if (score > best_score) {
            best_score = score;
            best_actions[0] = c;
            num_best = 1;
        } else if (score == best_score) {
            best_actions[num_best++] = c;
        }
    }

    /* If the position is solved (forced win/loss within search depth),
     * play the best move. Otherwise pick randomly among moves that are
     * within a small margin of the best heuristic score — still smart
     * but not deterministic. */
    if (best_score >= 10000 || best_score <= -10000) {
        /* Solved position — play optimally */
        if (num_best <= 1) return best_actions[0];
        return best_actions[rand() % num_best];
    }

    /* Unsolved — collect all moves within a margin of the best score */
    int margin = 3;  /* heuristic score tolerance */
    int good_actions[COLS];
    int num_good = 0;

    for (int c = 0; c < COLS; c++) {
        int r = get_drop_row(board, c);
        if (r < 0) continue;
        board[r][c] = player;
        int score = minimax(board, depth - 1, INT_MIN, INT_MAX, 0, player);
        board[r][c] = 0;
        if (score >= best_score - margin) {
            good_actions[num_good++] = c;
        }
    }

    if (num_good == 0) return best_actions[0];
    return good_actions[rand() % num_good];
}

/*
 * minimax_get_scores: compute minimax score for each column.
 *
 * board_flat: row-major int array of size 42
 * depth:      search depth
 * player:     current player (1 or 2)
 * scores:     output array of 7 ints (INT_MIN for illegal columns)
 *
 * Returns: 1 if any column has a solved score (|score| >= 10000), 0 otherwise
 */
int minimax_get_scores(int *board_flat, int depth, int player, int *scores) {
    int board[ROWS][COLS];
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            board[r][c] = board_flat[r * COLS + c];

    int has_solved = 0;
    for (int c = 0; c < COLS; c++) {
        int r = get_drop_row(board, c);
        if (r < 0) {
            scores[c] = INT_MIN;
            continue;
        }
        board[r][c] = player;
        scores[c] = minimax(board, depth - 1, INT_MIN, INT_MAX, 0, player);
        board[r][c] = 0;
        if (scores[c] >= 10000 || scores[c] <= -10000)
            has_solved = 1;
    }
    return has_solved;
}

/* ---- batch API with pthreads ---- */

typedef struct {
    int *board_flat;   /* pointer to this board's 42 ints */
    int depth;
    int player;
    int result;        /* output: best column */
    int *scores;       /* output: 7 scores (NULL if not needed) */
    int is_solved;     /* output: 1 if position has a forced win/loss */
} BatchTask;

static void *_batch_worker(void *arg) {
    BatchTask *task = (BatchTask *)arg;
    if (task->scores != NULL) {
        task->is_solved = minimax_get_scores(
            task->board_flat, task->depth, task->player, task->scores);
        /* Also compute best action from scores */
        int best = INT_MIN, best_col = -1;
        int board[ROWS][COLS];
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++)
                board[r][c] = task->board_flat[r * COLS + c];
        for (int c = 0; c < COLS; c++) {
            if (board[0][c] != 0) continue;
            if (task->scores[c] > best) {
                best = task->scores[c];
                best_col = c;
            }
        }
        task->result = best_col;
    } else {
        task->result = minimax_best_action(task->board_flat, task->depth, task->player);
        task->is_solved = 0;
    }
    return NULL;
}

/*
 * boards_flat: N boards concatenated, each 42 ints (row-major)
 * depths:      array of N depths
 * players:     array of N players
 * results:     output array of N best columns
 * n:           number of boards
 */
void minimax_batch(int *boards_flat, int *depths, int *players,
                   int *results, int n) {
    pthread_t *threads = (pthread_t *)malloc(n * sizeof(pthread_t));
    BatchTask *tasks = (BatchTask *)malloc(n * sizeof(BatchTask));

    for (int i = 0; i < n; i++) {
        tasks[i].board_flat = boards_flat + i * ROWS * COLS;
        tasks[i].depth = depths[i];
        tasks[i].player = players[i];
        tasks[i].result = -1;
        tasks[i].scores = NULL;
        tasks[i].is_solved = 0;
        pthread_create(&threads[i], NULL, _batch_worker, &tasks[i]);
    }

    for (int i = 0; i < n; i++) {
        pthread_join(threads[i], NULL);
        results[i] = tasks[i].result;
    }

    free(threads);
    free(tasks);
}

/*
 * minimax_batch_scores: like minimax_batch but also returns per-column scores.
 *
 * boards_flat:  N boards concatenated, each 42 ints
 * depths:       array of N depths
 * players:      array of N players
 * all_scores:   output array of N*7 ints (scores for each column per board)
 * solved_flags: output array of N ints (1 if solved, 0 otherwise)
 * n:            number of boards
 */
void minimax_batch_scores(int *boards_flat, int *depths, int *players,
                          int *all_scores, int *solved_flags, int n) {
    pthread_t *threads = (pthread_t *)malloc(n * sizeof(pthread_t));
    BatchTask *tasks = (BatchTask *)malloc(n * sizeof(BatchTask));

    for (int i = 0; i < n; i++) {
        tasks[i].board_flat = boards_flat + i * ROWS * COLS;
        tasks[i].depth = depths[i];
        tasks[i].player = players[i];
        tasks[i].result = -1;
        tasks[i].scores = all_scores + i * COLS;
        tasks[i].is_solved = 0;
        pthread_create(&threads[i], NULL, _batch_worker, &tasks[i]);
    }

    for (int i = 0; i < n; i++) {
        pthread_join(threads[i], NULL);
        solved_flags[i] = tasks[i].is_solved;
    }

    free(threads);
    free(tasks);
}
