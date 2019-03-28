/* Bounded Variable Least Sqaures Solver
 *
 * Copyright David Wiltshire (c), 2014
 * All rights reserved
 *
 * Licence: see file LICENSE
 */

#ifndef BVLS_H_
#define BVLS_H_

/** Bounded Variable Least Squares Solver.
 *
 * Solves the problem:
 *
 * 	     minimize ||Ax - b||
 *		x		2
 *	     with lb <= x <= ub
 *
 * Where
 *	A is an n x m matrix
 *	b is a m vector
 *	lb, ub are n vectors
 *	x is an n vector
 *
 * @param m [in] number of columns
 * @param n [in] number of rows
 * @param A [in] coefficient Matrix
 * @param b [in] target vector
 * @param lb [in] lower bounds
 * @param ub [in] upper bounds
 * @param x [out] result vector
 *
 * @return 0 if successfuly, < 0 on error
 */
int bvls(int m, int n,
	 const double *A,
	 const double *b,
	 const double *lb,
	 const double *ub,
	 double *x);

#endif // BVLS_H_
