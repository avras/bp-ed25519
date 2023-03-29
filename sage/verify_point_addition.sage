import random

q = 2**255 - 19
# Pallas base field order (native field order)
nf_order = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
# Vesta base field order (native field order)
# nf_order = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
width = 64 # Limb width
Fn = GF(nf_order)
Fq = GF(q)

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

def calc_native_limbs(a, num_limbs):
    a = int(a) # convert field element to big integer
    base = 1 << width
    limbs = []
    for i in range(num_limbs):
        limbs.append(Fn(a % base)) # limbs are native field elements
        a = a // base
    return limbs

def native_limbs_to_nonnative_field_element(a_l):
    base = 1
    a = Fq(0)
    for i in range(len(a_l)):
        a += Fq(base) * Fq(int(a_l[i]))
        base = base * (1 << width)
    return a

def native_limbs_to_bigint(a_l):
    base = 1
    a = 0
    for i in range(len(a_l)):
        a += base * int(a_l[i])
        base = base * (1 << width)
    return a

def calc_cubic_native_limbs_unreduced(a_l, b_l, c_l):
    g_sub = cubic_term_subscripts(4)
    g_l = [0]*len(g_sub)
    for m in range(len(g_l)):
        for (i,j,k) in g_sub[m]:
            g_l[m] += a_l[i]*b_l[j]*c_l[k]
    return g_l

def calc_quadratic_native_limbs(a_l, b_l):
    ab_sub = quadratic_term_subscripts_unequal_lengths(len(a_l), len(b_l))
    ab_l = [Fn(0)]*len(ab_sub)
    for m in range(len(ab_l)):
        for (i,j) in ab_sub[m]:
            ab_l[m] += a_l[i]*b_l[j]
    return ab_l

def fold_cubic_native_limbs_unreduced(g_l):
    assert(len(g_l) == 10)
    h_l = [0]*4
    h_l[0] = g_l[0]+38*g_l[4]+1444*g_l[8]
    h_l[1] = g_l[1]+38*g_l[5]+1444*g_l[9]
    h_l[2] = g_l[2]+38*g_l[6]
    h_l[3] = g_l[3]+38*g_l[7]
    return h_l

def fold_quadratic_native_limbs_unreduced(f_l):
    assert(len(f_l) == 7)
    h_l = [0]*7
    h_l[0] = f_l[0]+38*f_l[4]
    h_l[1] = f_l[1]+38*f_l[5]
    h_l[2] = f_l[2]+38*f_l[6]
    h_l[3] = f_l[3]
    return h_l

def add_native_limbed_ints_unequal_lengths(a_l, b_l):
    sum_len = max(len(a_l), len(b_l))
    sum_l = [Fn(0)]*sum_len
    for i in range(sum_len):
        if i < len(a_l):
            sum_l[i] += a_l[i]
        if i < len(b_l):
            sum_l[i] += b_l[i]
    return sum_l

def sub_native_limbed_ints_unequal_lengths(a_l, b_l):
    diff_len = max(len(a_l), len(b_l))
    diff_l = [Fn(0)]*diff_len
    for i in range(diff_len):
        if i < len(a_l):
            diff_l[i] += a_l[i]
        if i < len(b_l):
            diff_l[i] -= b_l[i]
    return diff_l

def check_native_limbed_ints_difference_is_zero_unequal_lengths(a_l, b_l):
    diff_len = max(len(a_l), len(b_l))
    diff_l = [0]*diff_len
    for i in range(diff_len):
        if i < len(a_l):
            diff_l[i] += a_l[i]
        if i < len(b_l):
            diff_l[i] -= b_l[i]
    
    base = 1 << width
    carries = [0]*(diff_len-1)
    for i in range(diff_len-1):
        if i == 0:
            carries[0] = (diff_l[0] // base)
            assert(diff_l[0]  == carries[0] * base)
            # print(carries[0])
        else:
            carries[i] = (carries[i-1] + diff_l[i]) // base
            assert(carries[i-1] + diff_l[i] == carries[i] * base)
            # print(carries[i])
    assert(diff_l[diff_len-1] + carries[diff_len-2] == 0)
    # print(diff_l)

    return diff_l

def verify_cubic_product(a, b, c, prod):
    r = a*b*c
    assert(r == prod)
    a_l = calc_native_limbs(a, 4)
    b_l = calc_native_limbs(b, 4)
    c_l = calc_native_limbs(c, 4)

    g_l = calc_cubic_native_limbs_unreduced(a_l, b_l, c_l)
    assert(r == native_limbs_to_nonnative_field_element(g_l))
    
    h_l_unreduced = fold_cubic_native_limbs_unreduced(g_l)
    h = native_limbs_to_nonnative_field_element(h_l_unreduced)
    assert(h == r)

    h_bigint = native_limbs_to_bigint(h_l_unreduced)
    r_bigint = int(r)

    t_bigint = (h_bigint - r_bigint) // q # quotient
    assert(t_bigint < 2**140)
    assert(h_bigint == t_bigint*q+r_bigint)

    t_l = calc_native_limbs(t_bigint, 3)
    q_l = calc_native_limbs(q, 4)
    r_l = calc_native_limbs(r_bigint, 4)

    tq_l = calc_quadratic_native_limbs(t_l, q_l)
    assert(t_bigint*q == native_limbs_to_bigint(tq_l))

    tq_plus_r_l_unreduced = add_native_limbed_ints_unequal_lengths(tq_l, r_l)
    assert(native_limbs_to_bigint(tq_plus_r_l_unreduced) == t_bigint*q+r_bigint)

    return check_native_limbed_ints_difference_is_zero_unequal_lengths(h_l_unreduced, tq_plus_r_l_unreduced)


def verify_x_coordinate_quadratic_is_zero(x1, x2, y1, y2, x3, v):
    x1_l = calc_native_limbs(x1, 4)
    x2_l = calc_native_limbs(x2, 4)
    x3_l = calc_native_limbs(x3, 4)
    y1_l = calc_native_limbs(y1, 4)
    y2_l = calc_native_limbs(y2, 4)
    v_l = calc_native_limbs(v, 4)
    q_l = calc_native_limbs(q, 4)
    q70_l = [0]*4
    for i in range(4):
        q70_l[i] = q_l[i] << 70

    x1y2_l = fold_quadratic_native_limbs_unreduced(calc_quadratic_native_limbs(x1_l, y2_l))
    x2y1_l = fold_quadratic_native_limbs_unreduced(calc_quadratic_native_limbs(x2_l, y1_l))
    x3v_l = fold_quadratic_native_limbs_unreduced(calc_quadratic_native_limbs(x3_l, v_l))
    quad_l = add_native_limbed_ints_unequal_lengths(x1y2_l, x2y1_l)
    quad_l = sub_native_limbed_ints_unequal_lengths(quad_l, x3_l)
    quad_l = sub_native_limbed_ints_unequal_lengths(quad_l, x3v_l)
    g_l = add_native_limbed_ints_unequal_lengths(quad_l, q70_l)

    g = native_limbs_to_bigint(g_l)
    assert(g > 0)
    assert(g % q == 0)
    t = g // q # quotient
    assert(t < 2**72)
    assert(g == t*q)

    t_l = calc_native_limbs(t, 2)
    tq_l = calc_quadratic_native_limbs(t_l, q_l)
    assert(t*q == native_limbs_to_bigint(tq_l))
    return check_native_limbed_ints_difference_is_zero_unequal_lengths(g_l, tq_l)

def verify_y_coordinate_quadratic_is_zero(x1, x2, y1, y2, y3, v):
    x1_l = calc_native_limbs(x1, 4)
    x2_l = calc_native_limbs(x2, 4)
    y1_l = calc_native_limbs(y1, 4)
    y2_l = calc_native_limbs(y2, 4)
    y3_l = calc_native_limbs(y3, 4)
    v_l = calc_native_limbs(v, 4)
    q_l = calc_native_limbs(q, 4)
    q70_l = [0]*4
    for i in range(4):
        q70_l[i] = q_l[i] << 70


    x1x2_l = fold_quadratic_native_limbs_unreduced(calc_quadratic_native_limbs(x1_l, x2_l))
    y1y2_l = fold_quadratic_native_limbs_unreduced(calc_quadratic_native_limbs(y1_l, y2_l))
    y3v_l = fold_quadratic_native_limbs_unreduced(calc_quadratic_native_limbs(y3_l, v_l))
    quad_l = add_native_limbed_ints_unequal_lengths(x1x2_l, y1y2_l)
    quad_l = add_native_limbed_ints_unequal_lengths(quad_l, y3v_l)
    quad_l = sub_native_limbed_ints_unequal_lengths(quad_l, y3_l)
    g_l = add_native_limbed_ints_unequal_lengths(quad_l, q70_l)

    g = native_limbs_to_bigint(g_l)
    assert(g > 0)
    assert(g % q == 0)
    t = g // q # quotient
    assert(t < 2**72)
    assert(g == t*q)

    t_l = calc_native_limbs(t, 2)
    tq_l = calc_quadratic_native_limbs(t_l, q_l)
    assert(t*q == native_limbs_to_bigint(tq_l))
    return check_native_limbed_ints_difference_is_zero_unequal_lengths(g_l, tq_l)

def ed25519_base_point():
    By = Fq(4) * ed25519_inv(5)
    Bx = ed25519_xrecover(By)
    return [Bx, By]

def ed25519_inv(x):
    return Fq(x)^(-1)

d = (-121665 * ed25519_inv(121666))

def ed25519_xrecover(y):
    I = pow(Fq(2),(q-1)//4)

    xx = Fq((y*y-1) * ed25519_inv(d*y*y+1))
    x = pow(xx,(q+3)/8)
    if (x*x - xx) != 0: x = (x*I) % q
    if int(x) % 2 != 0: x = -x
    return x

def ed25519_addition(P,Q):
    x1 = P[0]
    y1 = P[1]
    x2 = Q[0]
    y2 = Q[1]
    x3 = (x1*y2+x2*y1) * ed25519_inv(1+d*x1*x2*y1*y2)
    y3 = (y1*y2+x1*x2) * ed25519_inv(1-d*x1*x2*y1*y2)
    return [x3, y3]

def scalarmult(P,e):
    if e == 0: return [0,1]
    Q = scalarmult(P,e//2)
    Q = ed25519_addition(Q,Q)
    if e & 1: Q = ed25519_addition(Q,P)
    return Q

def verify_point_addition(x1, y1, x2, y2, x3, y3):
    u = d*x1*x2
    assert(verify_cubic_product(d, x1, x2, u))
    v = u*y1*y2
    assert(verify_cubic_product(u, y1, y2, v))
    assert(verify_x_coordinate_quadratic_is_zero(x1, x2, y1, y2, x3, v))
    assert(verify_y_coordinate_quadratic_is_zero(x1, x2, y1, y2, y3, v))
    return True


def main():
    a = random.randint(0, 1000)
    b = random.randint(0, 1000)

    B = ed25519_base_point()
    P = scalarmult(B, a)
    Q = scalarmult(B, b)
    R = ed25519_addition(P, Q)
    assert(verify_point_addition(P[0], P[1], Q[0], Q[1], R[0], R[1]))
    
if __name__ == "__main__":
    main()
