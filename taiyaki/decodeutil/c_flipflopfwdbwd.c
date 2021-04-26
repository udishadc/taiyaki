#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "c_flipflopfwdbwd.h"


/**
 *  Functions to calculate log-partition functions, forward and backward. The backward function is used in guiding beam search,
 *   and forward function is used for testing. 

 * The network outputs give a number w(b,t) for each block b and transition t, so w is an (nblocks x 40) matrix.
 * These are the transition weights 'score' below.
 * A path is a sequence of transitions p = [t0,t1,t2...]
 * The network outputs imply a score for each path, which is an un-normalised 'probability'  Q(p) = exp(-w(0,t0) - w(1,t1) - w(2,t2) ...)
 * The partition function is Z = sum_p Q(p) - sum is over all paths.
 * Forward and backward functions in this file calculate a matrix of restricted partition functions Z(b,t)
 * For example the backward partition function matrix B(b,t) = sum_p' Q(p') where the sum is over all paths covering blocks after b
 * and having transition t at block b.

**/


static float logsumexpf(float x, float y){
    float absdel = fabsf(x - y);
    //return fmaxf(x, y) + log1pf(exp(-(double)absdel));
    return fmaxf(x, y) + ((absdel < 17.0f) ? log1pf(expf(-absdel)) : 0.0f);
}



/**   Sum over all flip-flop paths backwards
 *
 *        From base (flip uppercase, flop lowercase)
 *      A C G T a c g t
 *  A   0     ---     7
 *  C   8     ---    15
 *  G  16     ---    23
 *  T  24     ---    31
 *  X  32     ---    39
 *
 *  X = lower(from)
 *
 *    nstate = nbase + nbase
 *    ntrans = nstate * (nbase + 1)
 *
 *    @param score      Array containing scores (nblock, ntrans)
 *    @param nbase      Number of bases
 *    @param nblock     Number of block (time-points) in score
 *    @param bwd [out]  Array for backwards scores (nblock + 1, nstate).  Last nstates
 *                         elements of array should be set to initial values on input.
 *
 *    @returns score
 **/
float flipflop_backward(const float * score, size_t nbase, size_t nblock, float * bwd){
    assert(NULL != score);
    assert(NULL != bwd);

    const size_t nstate = nbase + nbase;
    const size_t ntrans = nstate * (nbase + 1);


    for(size_t blk=nblock ; blk > 0 ; blk--){
        const size_t blkm1 = blk - 1;
        const float * pbwd = bwd + blk * nstate;
        float * cbwd = bwd + blkm1 * nstate;
        const float * cscore = score + blkm1 * ntrans;

        for(size_t b=0 ; b < nbase ; b++){
            //  Scores going to flop base
            cbwd[b] = cscore[nstate * nbase + b] + pbwd[nbase + b];
            cbwd[b + nbase] = cscore[nstate * nbase + b + nbase] + pbwd[nbase + b];
        }

        for(size_t to_base=0 ; to_base < nbase ; to_base++){
            //  Score going to flip base
            for(size_t from_state=0 ; from_state  < nstate ; from_state++){
                float sc = cscore[to_base * nstate + from_state] + pbwd[to_base];
                cbwd[from_state] = logsumexpf(cbwd[from_state], sc);
            }
        }
    }

    float total_score = bwd[0];
    for(size_t i=1 ; i < nstate ; i++){
        total_score = logsumexpf(total_score, bwd[i]);
    }

    return total_score;
}


/**   Sum over all flip-flop paths forwards
 *
 *        From base (flip uppercase, flop lowercase)
 *      A C G T a c g t
 *  A   0     ---     7
 *  C   8     ---    15
 *  G  16     ---    23
 *  T  24     ---    31
 *  X  32     ---    39
 *
 *  X = lower(from)
 *
 *    nstate = nbase + nbase
 *    ntrans = nstate * (nbase + 1)
 *
 *    @param score      Array containing scores (nblock, ntrans)
 *    @param nbase      Number of bases
 *    @param nblock     Number of block (time-points) in score
 *    @param fwd [out]  Array for forwards scores (nblock + 1, nstate).  First nstate
 *                         elements of array should be set to initial values on input.
 *
 *    @returns score
 **/
float flipflop_forward(const float * score, size_t nbase, size_t nblock, float * fwd){
    assert(NULL != score);
    assert(NULL != fwd);

    const size_t nstate = nbase + nbase;
    const size_t ntrans = nstate * (nbase + 1);

    for(size_t blk=0 ; blk < nblock ; blk++){
        const size_t blkp1 = blk + 1;
        const float * pfwd = fwd + blk * nstate;
        float *  cfwd = fwd + blkp1 * nstate;
        const float * cscore = score + blk * ntrans;

        for(size_t b=0 ; b < nbase ; b++){
            //  Scores to flop base (from flip and flop)
            const float flip_score = cscore[nstate * nbase + b] + pfwd[b];
            const float flop_score = cscore[nstate * nbase + b + nbase] + pfwd[b + nbase];
            cfwd[b + nbase] = logsumexpf(flip_score, flop_score);
        }

        for(size_t to_base=0 ; to_base < nbase ; to_base++){
            //  Score going to flip base
            cfwd[to_base] = cscore[to_base * nstate + 0] + pfwd[0];
            for(size_t from_state=1 ; from_state  < nstate ; from_state++){
                float sc = cscore[to_base * nstate + from_state] + pfwd[from_state];
                cfwd[to_base] = logsumexpf(cfwd[to_base], sc);
            }
        }
    }

    float total_score = fwd[nblock * nstate + 0];
    for(size_t i=1 ; i < nstate ; i++){
        total_score = logsumexpf(total_score, fwd[nblock * nstate + i]);
    }

    return total_score;
}
