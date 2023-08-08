import random
from ed25519 import *

# Pallas base field order (non-native field order)
nnf_order = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
# Vesta base field order (non-native field order)
# nnf_order = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
width = 64 # Limb width

def cubic_term_subscripts(num_limbs):
    max_limb_degree_in_cubic = 3*(num_limbs-1)
    subscript_sets = []
    for i in range(max_limb_degree_in_cubic+1):
        subscript_sets.append(set())

    for i in range(num_limbs):
        for j in range(num_limbs):
            for k in range(num_limbs):
                subscript_sets[i+j+k].add((i,j,k))

    return subscript_sets

def quadratic_term_subscripts_unequal_lengths(num_limbs1, num_limbs2):
    max_limb_degree_in_quadratic = num_limbs1+num_limbs2-2
    subscript_sets = []
    for i in range(max_limb_degree_in_quadratic+1):
        subscript_sets.append(set())

    for i in range(num_limbs1):
        for j in range(num_limbs2):
                subscript_sets[i+j].add((i,j))

    return subscript_sets

def calc_limbs(a, num_limbs):
    base = 1 << width
    limbs = []
    for i in range(num_limbs):
        limbs.append(a % base)
        a = a // base
    return limbs

def limbs_to_int(a_l):
    base = 1
    a = 0
    for i in range(len(a_l)):
        a += base * a_l[i]
        base = base * (1 << width)
    return a

def calc_cubic_limbs(a_l, b_l, c_l):
    g_sub = cubic_term_subscripts(4)
    g_l = [0]*len(g_sub)
    for m in range(len(g_l)):
        for (i,j,k) in g_sub[m]:
            g_l[m] += a_l[i]*b_l[j]*c_l[k]
    return g_l

def calc_quadratic_limbs(a_l, b_l):
    ab_sub = quadratic_term_subscripts_unequal_lengths(len(a_l), len(b_l))
    ab_l = [0]*len(ab_sub)
    for m in range(len(ab_l)):
        for (i,j) in ab_sub[m]:
            ab_l[m] += a_l[i]*b_l[j]
    return ab_l

def fold_cubic_limbs(g_l):
    assert(len(g_l) == 10)
    h_l = [0]*4
    h_l[0] = g_l[0]+38*g_l[4]+1444*g_l[8]
    h_l[1] = g_l[1]+38*g_l[5]+1444*g_l[9]
    h_l[2] = g_l[2]+38*g_l[6]
    h_l[3] = g_l[3]+38*g_l[7]
    return h_l

def fold_quadratic_limbs(f_l):
    assert(len(f_l) == 7)
    h_l = [0]*4
    h_l[0] = f_l[0]+38*f_l[4]
    h_l[1] = f_l[1]+38*f_l[5]
    h_l[2] = f_l[2]+38*f_l[6]
    h_l[3] = f_l[3]
    return h_l

def add_ints_unequal_lengths(a_l, b_l):
    sum_len = max(len(a_l), len(b_l))
    sum_l = [0]*sum_len
    for i in range(sum_len):
        if i < len(a_l):
            sum_l[i] += a_l[i]
        if i < len(b_l):
            sum_l[i] += b_l[i]
    return sum_l

def sub_ints_unequal_lengths(a_l, b_l):
    diff_len = max(len(a_l), len(b_l))
    diff_l = [0]*diff_len
    for i in range(diff_len):
        if i < len(a_l):
            diff_l[i] += a_l[i]
        if i < len(b_l):
            diff_l[i] -= b_l[i]
    return diff_l

def check_difference_is_zero_unequal_lengths(a_l, b_l):
    diff_l = sub_ints_unequal_lengths(a_l, b_l)
    return check_int_is_zero(diff_l)

def check_int_is_zero(a_l):
    a_len = len(a_l)
    
    base = 1 << width
    carries = [0]*(a_len-1)
    for i in range(a_len-1):
        if i == 0:
            carries[0] = a_l[0] // base
            assert(a_l[0]  == carries[0] *base)
            # print(carries[0])
        else:
            carries[i] = (carries[i-1] + a_l[i]) // base
            assert(carries[i-1] + a_l[i] == carries[i] * base)
            # print(carries[i])
    return a_l[a_len-1] + carries[a_len-2] == 0

def verify_cubic_product(a, b, c, prod):
    abc = a*b*c
    r = abc % q # remainder
    assert(r == prod)
    a_l = calc_limbs(a, 4)
    b_l = calc_limbs(b, 4)
    c_l = calc_limbs(c, 4)

    g_l = calc_cubic_limbs(a_l, b_l, c_l)
    assert(abc == limbs_to_int(g_l))

    h_l = fold_cubic_limbs(g_l)
    h = limbs_to_int(h_l)
    assert(h % q == r)

    t = (h - r) // q # quotient
    assert(t < 2**141)
    assert(h == t*q+r)

    t_l = calc_limbs(t, 3)
    q_l = calc_limbs(q, 4)
    r_l = calc_limbs(r, 4)

    tq_l = calc_quadratic_limbs(t_l, q_l)
    assert(t*q == limbs_to_int(tq_l))

    tq_plus_r_l = add_ints_unequal_lengths(tq_l, r_l)
    assert(limbs_to_int(tq_plus_r_l) == t*q+r)

    return check_difference_is_zero_unequal_lengths(h_l, tq_plus_r_l)

def verify_x_coordinate_quadratic_is_zero(x1, x2, y1, y2, x3, v):
    x1_l = calc_limbs(x1, 4)
    x2_l = calc_limbs(x2, 4)
    x3_l = calc_limbs(x3, 4)
    y1_l = calc_limbs(y1, 4)
    y2_l = calc_limbs(y2, 4)
    v_l = calc_limbs(v, 4)
    q_l = calc_limbs(q, 4)
    q71_l = [0]*4
    for i in range(4):
        q71_l[i] = q_l[i] << 71

    x1y2_l = fold_quadratic_limbs(calc_quadratic_limbs(x1_l, y2_l))
    x2y1_l = fold_quadratic_limbs(calc_quadratic_limbs(x2_l, y1_l))
    x3v_l = fold_quadratic_limbs(calc_quadratic_limbs(x3_l, v_l))
    quad_l = add_ints_unequal_lengths(x1y2_l, x2y1_l)
    quad_l = sub_ints_unequal_lengths(quad_l, x3_l)
    quad_l = sub_ints_unequal_lengths(quad_l, x3v_l)
    g_l = add_ints_unequal_lengths(quad_l, q71_l)

    g = limbs_to_int(g_l)
    assert(g > 0)
    assert(g % q == 0)
    t = g // q # quotient
    assert(t < 2**72)
    assert(g == t*q)

    t_l = calc_limbs(t, 2)
    tq_l = calc_quadratic_limbs(t_l, q_l)
    assert(t*q == limbs_to_int(tq_l))
    return check_difference_is_zero_unequal_lengths(g_l, tq_l)

def verify_y_coordinate_quadratic_is_zero(x1, x2, y1, y2, y3, v):
    x1_l = calc_limbs(x1, 4)
    x2_l = calc_limbs(x2, 4)
    y1_l = calc_limbs(y1, 4)
    y2_l = calc_limbs(y2, 4)
    y3_l = calc_limbs(y3, 4)
    v_l = calc_limbs(v, 4)
    q_l = calc_limbs(q, 4)
    q71_l = [0]*4
    for i in range(4):
        q71_l[i] = q_l[i] << 71


    x1x2_l = fold_quadratic_limbs(calc_quadratic_limbs(x1_l, x2_l))
    y1y2_l = fold_quadratic_limbs(calc_quadratic_limbs(y1_l, y2_l))
    y3v_l = fold_quadratic_limbs(calc_quadratic_limbs(y3_l, v_l))
    quad_l = add_ints_unequal_lengths(x1x2_l, y1y2_l)
    quad_l = add_ints_unequal_lengths(quad_l, y3v_l)
    quad_l = sub_ints_unequal_lengths(quad_l, y3_l)
    g_l = add_ints_unequal_lengths(quad_l, q71_l)

    g = limbs_to_int(g_l)
    assert(g > 0)
    assert(g % q == 0)
    t = g // q # quotient
    assert(t < 2**72)
    assert(g == t*q)

    t_l = calc_limbs(t, 2)
    tq_l = calc_quadratic_limbs(t_l, q_l)
    assert(t*q == limbs_to_int(tq_l))
    return check_difference_is_zero_unequal_lengths(g_l, tq_l)

def verify_point_addition(x1, y1, x2, y2, x3, y3):
    u = d*x1*x2 % q
    assert(verify_cubic_product(d, x1, x2, u))
    v = u*y1*y2 % q
    assert(verify_cubic_product(u, y1, y2, v))
    assert(verify_x_coordinate_quadratic_is_zero(x1, x2, y1, y2, x3, v))
    assert(verify_y_coordinate_quadratic_is_zero(x1, x2, y1, y2, y3, v))
    return True

def main():
    a = random.randint(0, 1000)
    b = random.randint(0, 1000)

    P = scalarmult(B, a)
    Q = scalarmult(B, b)
    R = edwards(P, Q)

    assert(verify_point_addition(P[0], P[1], Q[0], Q[1], R[0], R[1]))

if __name__ == "__main__":
    main()
