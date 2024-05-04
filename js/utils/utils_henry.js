
import {mul_matrix_scalar, mul_matrix_matrix, sub_matrix_matrix, add_matrix_matrix, transpose} from "./utils_math.js";

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

export function sqrtpi2(x) {
    return Math.PI * Math.sqrt(0.5*(x+1));
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

export function hat(v) {
    return [[0, -v[2], v[1]],
        [v[2],0,-v[0]],
        [-v[1], v[0],0]];
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

