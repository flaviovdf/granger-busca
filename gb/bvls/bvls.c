/* Bounded Variable Least Squares
 *
 * Solves the Bounded Variable Least Squares problem (or box-constrained
 * least squares) of:
 *	minimize   ||Ax - b||
 *	   x		     2
 *
 *	given lb <= x <= ub
 *
 *	where:
 *		A is an m x n matrix
 *		b, x, lb, ub are n vectors
 *
 * Based on the article Stark and Parker "Bounded-Variable Least Squares: an
 * Alogirthm and Applications" retrieved from:
 * http://www.stat.berkeley.edu/~stark/Preprints/bvls.pdf
 *
 * Copyright David Wiltshire (c), 2014
 * All rights reserved
 *
 * Licence: see file LICENSE
 */

// Local Includes
#include "bvls.h"

// Third Party Includes
#include "cblas.h"
#include "lapacke.h"

// Standard Library Includes
#include <assert.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

struct problem {
	const double *A;
	const double *b;
	const double *lb;
	const double *ub;
	const int m;
	const int n;
};

struct work {
	int *indices;
	int *istate;
	double *w;
	double *s;
	double *A;
	double *z;
	int num_free;
	int prev;
	int rank;
};

/* Computes w(*) = trans(A)(Ax -b), the negative gradient of the residual.*/
static void
negative_gradient(struct problem *prob, struct work *work, double *x)
{
	int m = prob->m;
	int n = prob->n;
	// z = b
	memcpy(work->z, prob->b, m * sizeof(*work->z));
	// z = (-Ax + b) = (b - Ax)
	cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, -1.0, prob->A, m, x, 1, 1.0, work->z, 1);
	// w = trans(A)z + 0w = trans(A)r
	cblas_dgemv(CblasColMajor, CblasTrans, m, n, 1.0, prob->A, m, work->z, 1, 0.0, work->w, 1);
}

/* Find the index which most wants to be free.  Or return -1 */
static int
find_index_to_free(int n, struct work *work)
{
	int index = -1;
	double max_grad = 0.0;

	for (int i = 0; i < n; ++i) {
		double gradient = -work->w[i] * work->istate[i];
		if (gradient > max_grad) {
			max_grad = gradient;
			index = i;
		}
	}
	return index;
}

/* Move index to the free set */
static void
free_index(int index, struct work *work)
{
	assert(index >= 0);
	work->istate[index] = 0;
	work->indices[work->num_free] = index;
	++(work->num_free);
}

/*
 * Build matrix A' and b' where A' is those columns of A that are free
 * and b' is the vector less the contribution of the bound variables
 */
static void
build_free_matrices(struct problem *prob, struct work *work, double *x)
{
	int m = prob->m;
	int n = prob->n;

	/* Set A' to free columns of A */
	for (int i = 0; i < work->num_free; ++i) {
		int ii = work->indices[i];
		memcpy((work->A + i * m), (prob->A + m * ii), m * sizeof(*work->A));
	}

	/* Set b' = b */
	memcpy(work->z, prob->b, m * sizeof(*work->z));
	/* Adjust b'j = bj - sum{Aij * x[j]} for i not in Free set */
	for (int i = 0; i < m; ++i)  {
		for (int j = 0; j < n; ++j) { 
			if (work->istate[j] != 0)
				work->z[i] -= *(prob->A + j * m + i) * x[j];
		}
	}
}

/* Check that suggested solution is with in bounds */
static bool
check_bounds(struct problem *prob, struct work *work)
{
	for (int i = 0; i < work->num_free; ++i) {
		int ii = work->indices[i];
		if (work->z[i] < prob->lb[ii] || work->z[i] > prob->ub[ii])
			return false;
	}
	return true;
}

/* Set variable to upper/lower bound */
static void
bind_index(int index, bool up, double fixed, struct work *work, double *x)
{
	x[work->indices[index]] = fixed;
	work->istate[work->indices[index]] = up ? 1 : -1;
	--(work->num_free);
	for (int i = index; i < work->num_free; ++i)
		work->indices[i] = work->indices[i + 1];
}

/* 
 * Find a variable to bind to a limit, and interpolate the solution:
 *	xj = xj + alpha(zj' - xj)
 */
static int
find_index_to_bind(struct problem *prob, struct work *work, double *x)
{
	int index = -1;
	bool bind_up = false;
	double alpha = DBL_MAX;

	for (int i = 0; i < work->num_free; ++i) {
		int ii = work->indices[i];
		double interpolate;
		if (work->z[i] <= prob->lb[ii]) {
			interpolate = (prob->lb[ii] - x[ii]) / (work->z[i] - x[ii]);
			if (interpolate < alpha) {
				alpha = interpolate;
				index = i;
				bind_up = false;
			}
		} else if (work->z[i] >= prob->ub[ii]) {
			interpolate = (prob->ub[ii] - x[ii]) / (work->z[i] - x[ii]);
			if (interpolate < alpha) {
				alpha = interpolate;
				index = i;
				bind_up = true;
			}
		}
	}

	assert(index >= 0);

	for (int i = 0; i < work->num_free; ++i) {
		int ii = work->indices[i];
		x[ii] += alpha * (work->z[i] - x[ii]);
	}

	int ii = work->indices[index];
	double limit = bind_up? prob->ub[ii] : prob->lb[ii];
	bind_index(index, bind_up, limit, work, x);
	return index;
}

/* Move variables that are out of bounds to their respective bound */
static void
adjust_sets(struct problem *prob, struct work *work, double *x)
{
	// We must repeat each loop where we bind an index as we will
	// have removed that entry from indices
	for (int i = 0; i < work->num_free; ++i) {
		int ii = work->indices[i];
		if (x[ii] <= prob->lb[ii]) {
			bind_index(i--, false, prob->lb[ii], work, x);
		} else if (x[i] >= prob->ub[i]) {
			bind_index(i--, true, prob->ub[ii], work, x);
		}
	}
}

/*
 * Check the result and adjust the sets accordingly.
 * Returns the index bound
 */
static int
check_result(struct problem *prob, struct work *work, double *x)
{
	int err = 0;
	if (check_bounds(prob, work)) {
		for (int i = 0; i < work->num_free; ++i)
			x[work->indices[i]] = work->z[i];
	} else {
		work->prev = find_index_to_bind(prob, work, x);
		adjust_sets(prob, work, x);
		err = -1;
	}
	return err;
}

/* Loop B in Lawson and Hanson.  Continually loop solving the problem and
 * binding at least one variable each step till we find a valid solution or
 * until every variable is bound.
 */
static void 
find_valid_result(struct problem *prob, struct work *work, double *x)
{
	int m = prob->m;
	int n = prob->n;
	int mn = m > n ? m : n;

	while (work->num_free > 0) {
		build_free_matrices(prob, work, x);
		int rc = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', m, work->num_free,
				1, work->A, m, work->z, mn);
		assert(rc == 0);
		if (check_result(prob, work, x) == 0)
			break;
	}
}

/* Set all variables to be bound to the lower bound */
static void
set_to_lower_bound(struct problem *prob, struct work *work, double *x)
{
	int m = prob->m;
	int n = prob->n;
	work->rank = m < n ? m : n;
	for (int i = 0; i < n; ++i) {
		x[i] = prob->lb[i];
		work->istate[i] = -1;
		work->indices[i] = -1;
	}
	work->num_free = 0;
}

/* Use SVD to solve the problem unconstrained.  If that fails to find a
 * solution within bounds then attempt to find a valid starting solution
 * with some variables free
 */
static int
solve_unconstrained(struct problem *prob, struct work *work, double *x)
{
	int m = prob->m;
	int n = prob->n;
	int rc;

	memcpy(work->A, prob->A, m * n * sizeof(*work->A));
	memcpy(work->z, prob->b, m * sizeof(*work->z));
	for (int i = m; i < n; ++i)
		work->z[i] = 0.0;
	rc = LAPACKE_dgelss(LAPACK_COL_MAJOR, m, n, 1, work->A, m, work->z,
			    m > n ? m : n, work->s, FLT_MIN, &work->rank);
	if (rc < 0) {
		set_to_lower_bound(prob, work, x);
		return -1;
	}

	for (int i = 0; i < work->num_free; ++i)
		x[work->indices[i]] = work->z[i];

	if (check_bounds(prob, work)) {
		rc = 0;
		return 0;
	} else {
		adjust_sets(prob, work, x);
		find_valid_result(prob, work, x);
		return -1;
	}
}

/* Allocate working arrays */
static int
allocate(int m, int n, struct work *work)
{
	int mn = (m > n ? m : n);

	if ((work->w	   = malloc(n * sizeof(*work->w))	) == NULL ||
	    (work->A	   = malloc(m * n * sizeof(*work->A))	) == NULL ||
	    (work->z	   = malloc(mn * sizeof(*work->z))	) == NULL ||
	    (work->s	   = malloc(mn * sizeof(*work->s))	) == NULL ||
	    (work->istate  = malloc(n * sizeof(*work->istate))	) == NULL ||
	    (work->indices = malloc(n * sizeof(*work->indices))	) == NULL)
		return -ENOMEM;
	else
		return 0;
}

/* Free memory */
static void
clean_up(struct work *work)
{
	free(work->w);
	free(work->A);
	free(work->z);
	free(work->s);
	free(work->istate);
	free(work->indices);
}

/*
 * Initializes the problems:
 *	- check that lb and ub are valid
 *	- add every element to the free set
 */
static int
init(struct problem *prob, struct work *work)
{
	int n = prob->n;

	work->prev = -1;
	work->num_free = n;
	for (int i = 0; i < n; ++i) {
		if (prob->lb[i] > prob->ub[i])
			return -1;
		work->indices[i] = i;
		work->istate[i] = 0;
	}
	return 0;
}

/* The BVLS main function */
int
bvls(int m, int n, const double *restrict A, const double *restrict b,
	const double *restrict lb, const double *restrict ub, double *restrict x)
{
	struct work work;
	struct problem prob = {.m = m, .n = n, .A = A, .b = b, .lb = lb, .ub = ub};
	int rc;
	int loops = 3 * n;

	rc = allocate(m, n, &work);
	if (rc < 0)
		goto out;
	rc = init(&prob, &work);
	if (rc < 0)
		goto out;

	if (solve_unconstrained(&prob, &work, x) == 0)
		goto out;

	negative_gradient(&prob, &work, x);
	for (loops = 3 * n; loops > 0; --loops) {
		int index_to_free = find_index_to_free(n, &work);
		/*
		 * If no index on a bound wants to move in to the
		 * feasible region then we are done
		 */
		if (index_to_free < 0)
			break;

		if (index_to_free == work.prev) {
			work.w[work.prev] = 0.0;
			continue;
		}

		/* Move index to free set */
		free_index(index_to_free, &work);
		/* Solve Problem for free set */
		build_free_matrices(&prob, &work, x);
		rc = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', m, work.num_free, 1,
				    work.A, m, work.z, m > n ? m : n);
		if (rc < 0) {
			work.prev = index_to_free;
			work.w[work.prev] = 0.0;
			if (x[work.prev] == prob.lb[work.prev]) {
				work.istate[work.prev] = -1;
			} else {
				work.istate[work.prev] = 1;
			}
			--work.num_free;
			continue;
		}

		if (check_result(&prob, &work, x) == 0) {
			work.prev = -1;
		} else {
			find_valid_result(&prob, &work, x);
		}
		if (work.num_free == work.rank)
			break;
		negative_gradient(&prob, &work, x);
	}
	if (loops == 0) // failed to converge
		rc = -1;
out:
	clean_up(&work);
	return rc;
}
