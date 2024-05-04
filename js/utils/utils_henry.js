
import {mul_matrix_scalar, mul_matrix_matrix, sub_matrix_matrix, add_matrix_matrix, transpose} from "./utils_math.js";

export function dot(x, y) {
    return x[0]*y[0]+x[1]*y[1]+x[2]*y[2];
}

export function sqrtpi2(x) {
    return Math.PI * Math.sqrt(0.5*(x+1));
}

export function sigmoid(x) {
    return 1 / (1+Math.exp(-x));
}


export function Rx(theta) {
    let sin = Math.sin(theta);
    let cos = Math.cos(theta);
    return [[1, 0, 0],
        [0,cos, -sin],
        [0,sin, cos]];
}

export function Ry(theta) {
    let sin = Math.sin(theta);
    let cos = Math.cos(theta);
    return [[cos, 0, sin],
        [0,1,0],
        [-sin,0, cos]];
}

export function Rz(theta) {
    let sin = Math.sin(theta);
    let cos = Math.cos(theta);
    return [[cos, -sin, 0],
        [sin,cos, 0],
        [0,0,1]];
}
export function so3_log(m) {
    let tr = m[0][0] + m[1][1] + m[2][2];
    let beta = Math.acos((tr-1)/2);

    if (Math.abs(beta) < 0.0001) {
        let mult = 0.5 + beta**2/12+7*beta**4/720;
        return mul_matrix_scalar(sub_matrix_matrix(m, transpose(m)), mult);
    } else if ((beta > Math.PI-0.0001) || (beta < -Math.PI+0.0001)) {
        return [[0,-sqrtpi2(m[2][2]), sqrtpi2(m[1][1])],
            [sqrtpi2(m[2][2]), 0, -sqrtpi2(m[0][0])],
            [-sqrtpi2(m[1][1]),sqrtpi2(m[0][0]),0]
        ];
    } else {
        let mult = beta / (2*Math.sin(beta));
        return mul_matrix_scalar(sub_matrix_matrix(m, transpose(m)), mult);
    }
}

export function so3_exp(m) {
    let a = m[2][1];
    let b = m[0][2];
    let c = m[1][0];
    let beta = Math.sqrt(a**2+b**2+c**2);
    let p = 0;
    let q = 0;
    if (beta < 0.0001) {
        p = 1 - beta**2/6+beta**4/120;
        q = 1/2-beta**2/24+beta**4/720;
    } else {
        p = Math.sin(beta)/beta;
        q = (1-Math.cos(beta))/beta**2;
    }
    let result = [[1,0,0],[0,1,0],[0,0,1]];
    result = add_matrix_matrix(result, mul_matrix_scalar(m, p));
    result = add_matrix_matrix(result, mul_matrix_scalar(mul_matrix_matrix(m, m),q));
    return result;
}

export function se3_log(m) {
    let SO3 = [[m[0][0], m[0][1],m[0][2]],
        [m[1][0], m[1][1],m[1][2]],
        [m[2][0], m[2][1],m[2][2]]];
    let so3 = so3_log(SO3);
    let beta = Math.sqrt(so3[0][1]**2+so3[0][2]**2+so3[1][2]**2);
    let p = 0;
    let q = 0;
    if (beta < 0.0001) {
        p = 0.5 - beta**2/24+beta**4/720;
        q = 1/6-beta**2/120+beta**4/5040;
    } else {
        p = (1-Math.cos(beta))/beta**2;
        q = (beta-Math.sin(beta))/beta**3;
    }
    let identity = [[1,0,0],[0,1,0],[0,0,1]];
    let term2 = mul_matrix_scalar(so3, p);
    let term3 = mul_matrix_scalar(mul_matrix_matrix(so3, so3), q);
    let M = add_matrix_matrix(identity, add_matrix_matrix(term2, term3));
    let Minv = matrix_inverse_3x3(M);
    let t = [[m[0][3]],[m[1][3]],[m[2][3]]];
    let s = vec(mul_matrix_matrix(Minv, t));

    return [[so3[0][0],so3[0][1],so3[0][2],s[0]],
        [so3[1][0],so3[1][1],so3[1][2],s[1]],
        [so3[2][0],so3[2][1],so3[2][2],s[2]],
        [0,0,0,0]];
}

export function se3_exp(m) {
    let so3 = [[m[0][0], m[0][1],m[0][2]],
        [m[1][0], m[1][1],m[1][2]],
        [m[2][0], m[2][1],m[2][2]]];
    let SO3 = so3_exp(so3);

    let a = m[2][1];
    let b = m[0][2];
    let c = m[1][0];
    let beta = Math.sqrt(a**2+b**2+c**2);
    let p = 0;
    let q = 0;
    if (beta < 0.0001) {
        p = 1/2 - beta**2/24+beta**4/720;
        q = 1/6-beta**2/120+beta**4/5040;
    } else {
        p = (1-Math.cos(beta))/beta**2;
        q = (beta-Math.sin(beta))/beta**3;
    }
    let M = [[1,0,0],[0,1,0],[0,0,1]];
    M = add_matrix_matrix(M, mul_matrix_scalar(so3, p));
    M = add_matrix_matrix(M, mul_matrix_scalar(mul_matrix_matrix(so3, so3),q));
    let s = vec(mul_matrix_matrix(M, [[m[0][3]],[m[1][3]],[m[2][3]]]));

    return [[SO3[0][0],SO3[0][1],SO3[0][2],s[0]],
        [SO3[1][0],SO3[1][1],SO3[1][2],s[1]],
        [SO3[2][0],SO3[2][1],SO3[2][2],s[2]],
        [0,0,0,1]];
}

export function se3_inv(se3) {
    let so3_inv = [[se3[0][0],se3[1][0],se3[2][0]],
        [se3[0][1],se3[1][1],se3[2][1]],
        [se3[0][2],se3[1][2],se3[2][2]]];
    let t = [[-se3[0][3]], [-se3[1][3]],[-se3[2][3]]];
    let t_inv = mul_matrix_matrix(so3_inv,t);
    return [[so3_inv[0][0],so3_inv[0][1],so3_inv[0][2],t_inv[0]],
        [so3_inv[1][0],so3_inv[1][1],so3_inv[1][2],t_inv[1]],
        [so3_inv[2][0],so3_inv[2][1],so3_inv[2][2],t_inv[2]],
        [0,0,0,1]];
}

export function vee(se3) {
    return [-se3[1][2],se3[0][2], -se3[0][1],
        se3[0][3],se3[1][3],se3[2][3]];
}

export function hat(r6) {
    return [[0, -r6[2], r6[1],r6[3]],
        [r6[2],0,-r6[0], r6[4]],
        [-r6[1], r6[0],0,r6[5]],
        [0,0,0,0]];
}
export function Txyz(x, y, z) {
    return [[1,0,0,x],
        [0,1,0,y],
        [0,0,1,z],
        [0,0,0,1]]
}

export function vec(a) {
    let b = a.map(element=>element[0]);
    return [b[0],b[1],b[2]];
}

// A must be a 3x3 matrix in row major order
// [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
export function matrix_inverse_3x3(A) {
    let det = A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2]) -
        A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
        A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    if (det === 0) {
        return null; // No inverse exists if determinant is 0
    }

    let cofactors = [
        [
            (A[1][1] * A[2][2] - A[2][1] * A[1][2]),
            -(A[1][0] * A[2][2] - A[1][2] * A[2][0]),
            (A[1][0] * A[2][1] - A[2][0] * A[1][1])
        ],
        [
            -(A[0][1] * A[2][2] - A[0][2] * A[2][1]),
            (A[0][0] * A[2][2] - A[0][2] * A[2][0]),
            -(A[0][0] * A[2][1] - A[2][0] * A[0][1])
        ],
        [
            (A[0][1] * A[1][2] - A[0][2] * A[1][1]),
            -(A[0][0] * A[1][2] - A[1][0] * A[0][2]),
            (A[0][0] * A[1][1] - A[1][0] * A[0][1])
        ]
    ];

    let adjugate = [
        [cofactors[0][0] / det, cofactors[1][0] / det, cofactors[2][0] / det],
        [cofactors[0][1] / det, cofactors[1][1] / det, cofactors[2][1] / det],
        [cofactors[0][2] / det, cofactors[1][2] / det, cofactors[2][2] / det]
    ];

    return adjugate;
}

export function dist_se4(m1, m2) {
    let disp = mul_matrix_matrix(se3_inv(m1), m2);
    let lndisp = se3_log(disp);

    let m4 = mul_matrix_matrix(m1, se3_exp(mul_matrix_scalar(lndisp, settings.t)));
    let vlndisp = vee(lndisp);
    let sum = 0;
    vlndisp.forEach(num => sum += num**2);
    return Math.sqrt(sum);
}