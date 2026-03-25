#include <stdlib.h>
#include <limits.h>
#include <string.h>

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
    if (me_count == 4) return 100;
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
    if (winner == me)  return  100 + depth;
    if (winner == opp) return -100 - depth;
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
    if (best_score >= 100 || best_score <= -100) {
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
