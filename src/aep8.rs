#![allow(unused_assignments, unused_mut, dead_code)]

include!("aep8_data.rs");

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Particle {
    Electron,
    Proton,
}

#[derive(Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub(crate) enum SolarCycle {
    Min,
    Max,
}

fn get_model(particle: Particle, solar: SolarCycle) -> (&'static [i32; 8], &'static [i32]) {
    match (particle, solar) {
        (Particle::Electron, SolarCycle::Min) => (&AE8MIN_DESCR, &AE8MIN_MAP),
        (Particle::Electron, SolarCycle::Max) => (&AE8MAX_DESCR, &AE8MAX_MAP),
        (Particle::Proton, SolarCycle::Min) => (&AP8MIN_DESCR, &AP8MIN_MAP),
        (Particle::Proton, SolarCycle::Max) => (&AP8MAX_DESCR, &AP8MAX_MAP),
    }
}

fn trara2(map: &[i32], start: usize, il: i32, ib: i32, fistep: f32) -> f32 {
    let fnl = il as f32;
    let fnb = ib as f32;
    let mut itime = 0;
    let mut i1: usize = 0;
    let mut i2: usize = 0;
    let mut l1: i32 = 0;
    let mut l2: i32;

    loop {
        l2 = map[start + i2];
        if map[start + i2 + 1] > il {
            break;
        }
        i1 = i2;
        l1 = l2;
        i2 += l2 as usize;
    }

    if l1 < 4 && l2 < 4 {
        return 0.0;
    }

    if map[start + i2 + 2] > map[start + i1 + 2] {
        std::mem::swap(&mut i1, &mut i2);
        std::mem::swap(&mut l1, &mut l2);
    }

    let fll1 = map[start + i1 + 1] as f32;
    let fll2 = map[start + i2 + 1] as f32;
    let dfl = (fnl - fll1) / (fll2 - fll1);
    let mut flog1 = map[start + i1 + 2] as f32;
    let mut flog2 = map[start + i2 + 2] as f32;
    let mut fkb1: f32 = 0.0;
    let mut fkb2: f32 = 0.0;

    if l1 < 4 {
        let fincr2 = map[start + i2 + 3] as f32;
        let flogm = flog1 + (flog2 - flog1) * dfl;
        let fkbm: f32 = 0.0;
        fkb2 += fincr2;
        flog2 -= fistep;
        let sl2 = flog2 / fkb2;
        let sl1: f32 = -900000.0;
        return interp_loop(
            map, start, fistep, fnb, dfl, &mut flog1, &mut flog2, &mut fkb1, &mut fkb2, flogm,
            fkbm, sl1, sl2, i1, i2, l1, l2, 4, 4,
        );
    }

    let mut j2: usize = 4;
    while j2 <= l2 as usize {
        let fincr2 = map[start + i2 + j2 - 1] as f32;
        if fkb2 + fincr2 > fnb {
            if itime == 1 {
                if j2 == 4 {
                    let flogm = flog1 + (flog2 - flog1) * dfl;
                    let fkbm: f32 = 0.0;
                    fkb2 += fincr2;
                    flog2 -= fistep;
                    let sl2 = flog2 / fkb2;
                    let j1: usize = 4;
                    let fincr1 = map[start + i1 + j1 - 1] as f32;
                    fkb1 += fincr1;
                    flog1 -= fistep;
                    let sl1 = flog1 / fkb1;
                    return interp_loop(
                        map, start, fistep, fnb, dfl, &mut flog1, &mut flog2, &mut fkb1, &mut fkb2,
                        flogm, fkbm, sl1, sl2, i1, i2, l1, l2, j1, j2,
                    );
                }
                let sl2 = flog2 / fkb2;
                let mut fkbj1: f32 = 0.0;
                let mut j1: usize = 4;
                while j1 <= l1 as usize {
                    let fincr1 = map[start + i1 + j1 - 1] as f32;
                    fkb1 += fincr1;
                    flog1 -= fistep;
                    fkbj1 = ((flog1 / fistep) * fincr1 + fkb1) / ((fincr1 / fistep) * sl2 + 1.0);
                    if fkbj1 <= fkb1 {
                        break;
                    }
                    j1 += 1;
                }
                if fkbj1 <= fkb2 {
                    let fkbm = fkbj1 + (fkb2 - fkbj1) * dfl;
                    let flogm = fkbm * sl2;
                    flog2 -= fistep;
                    fkb2 += fincr2;
                    let sl1 = flog1 / fkb1;
                    let sl2_new = flog2 / fkb2;
                    return interp_loop(
                        map, start, fistep, fnb, dfl, &mut flog1, &mut flog2, &mut fkb1, &mut fkb2,
                        flogm, fkbm, sl1, sl2_new, i1, i2, l1, l2, j1, j2,
                    );
                } else {
                    return 0.0;
                }
            }

            return init_interp(
                map, start, fistep, fnb, dfl, &mut flog1, &mut flog2, &mut fkb1, &mut fkb2, i1, i2,
                l1, l2,
            );
        }
        fkb2 += fincr2;
        flog2 -= fistep;
        j2 += 1;
    }

    itime += 1;
    if itime == 1 {
        std::mem::swap(&mut i1, &mut i2);
        std::mem::swap(&mut l1, &mut l2);
        flog1 = map[start + i1 + 2] as f32;
        flog2 = map[start + i2 + 2] as f32;
        fkb1 = 0.0;
        fkb2 = 0.0;
        j2 = 4;
        while j2 <= l2 as usize {
            let fincr2 = map[start + i2 + j2 - 1] as f32;
            if fkb2 + fincr2 > fnb {
                return init_interp(
                    map, start, fistep, fnb, dfl, &mut flog1, &mut flog2, &mut fkb1, &mut fkb2, i1,
                    i2, l1, l2,
                );
            }
            fkb2 += fincr2;
            flog2 -= fistep;
            j2 += 1;
        }
        return 0.0;
    }

    0.0
}

#[allow(clippy::too_many_arguments)]
fn init_interp(
    map: &[i32],
    start: usize,
    fistep: f32,
    fnb: f32,
    dfl: f32,
    flog1: &mut f32,
    flog2: &mut f32,
    fkb1: &mut f32,
    fkb2: &mut f32,
    i1: usize,
    i2: usize,
    l1: i32,
    l2: i32,
) -> f32 {
    *fkb1 = 0.0;
    *fkb2 = 0.0;
    let fincr2 = map[start + i2 + 3] as f32;
    *flog2 = map[start + i2 + 2] as f32;
    *flog1 = map[start + i1 + 2] as f32;
    let flogm = *flog1 + (*flog2 - *flog1) * dfl;
    let fkbm: f32 = 0.0;
    *fkb2 += fincr2;
    *flog2 -= fistep;
    let sl2 = *flog2 / *fkb2;
    if l1 < 4 {
        let sl1: f32 = -900000.0;
        return interp_loop(
            map, start, fistep, fnb, dfl, flog1, flog2, fkb1, fkb2, flogm, fkbm, sl1, sl2, i1, i2,
            l1, l2, 4, 4,
        );
    }
    let fincr1 = map[start + i1 + 3] as f32;
    *fkb1 += fincr1;
    *flog1 -= fistep;
    let sl1 = *flog1 / *fkb1;
    interp_loop(
        map, start, fistep, fnb, dfl, flog1, flog2, fkb1, fkb2, flogm, fkbm, sl1, sl2, i1, i2, l1,
        l2, 4, 4,
    )
}

#[allow(clippy::too_many_arguments)]
fn interp_loop(
    map: &[i32],
    start: usize,
    fistep: f32,
    fnb: f32,
    dfl: f32,
    flog1: &mut f32,
    flog2: &mut f32,
    fkb1: &mut f32,
    fkb2: &mut f32,
    mut flogm: f32,
    mut fkbm: f32,
    mut sl1: f32,
    mut sl2: f32,
    i1: usize,
    i2: usize,
    l1: i32,
    l2: i32,
    mut j1: usize,
    mut j2: usize,
) -> f32 {
    loop {
        if sl1 < sl2 {
            if j2 > l2 as usize {
                return 0.0;
            }
            let fincr2 = map[start + i2 + j2 - 1] as f32;
            let fkbj2 = ((*flog2 / fistep) * fincr2 + *fkb2) / ((fincr2 / fistep) * sl1 + 1.0);
            let fkb = *fkb1 + (fkbj2 - *fkb1) * dfl;
            let flog = fkb * sl1;
            if fkb >= fnb {
                if fkb < fkbm + 1e-10 {
                    return 0.0;
                }
                return (flogm + (flog - flogm) * ((fnb - fkbm) / (fkb - fkbm))).max(0.0);
            }
            fkbm = fkb;
            flogm = flog;
            if j1 as i32 >= l1 {
                return 0.0;
            }
            j1 += 1;
            let fincr1 = map[start + i1 + j1 - 1] as f32;
            *flog1 -= fistep;
            *fkb1 += fincr1;
            sl1 = *flog1 / *fkb1;
        } else {
            if j1 > l1 as usize {
                return 0.0;
            }
            let fincr1 = map[start + i1 + j1 - 1] as f32;
            let fkbj1 = ((*flog1 / fistep) * fincr1 + *fkb1) / ((fincr1 / fistep) * sl2 + 1.0);
            let fkb = fkbj1 + (*fkb2 - fkbj1) * dfl;
            let flog = fkb * sl2;
            if fkb >= fnb {
                if fkb < fkbm + 1e-10 {
                    return 0.0;
                }
                return (flogm + (flog - flogm) * ((fnb - fkbm) / (fkb - fkbm))).max(0.0);
            }
            fkbm = fkb;
            flogm = flog;
            if j2 as i32 >= l2 {
                return 0.0;
            }
            j2 += 1;
            let fincr2 = map[start + i2 + j2 - 1] as f32;
            *flog2 -= fistep;
            *fkb2 += fincr2;
            sl2 = *flog2 / *fkb2;
        }
    }
}

fn trara1(descr: &[i32; 8], map: &[i32], fl: f64, bb0: f64, energy: f64) -> f64 {
    let fistep = descr[6] as f32 / descr[1] as f32;
    let escale = descr[3] as f32;
    let fscale = descr[6] as f32;
    let xnl = (fl.abs().min(15.6)) as f32;
    let nl = (xnl * descr[4] as f32) as i32;
    let bb0_clamped = bb0.max(1.0) as f32;
    let nb = ((bb0_clamped - 1.0) * descr[5] as f32) as i32;

    let mut i0: usize = 0;
    let mut i1: usize = 0;
    let mut i2: usize = map[0] as usize;
    let mut i3 = i2 + map[i2] as usize;
    let mut l3 = map[i3];
    let mut e0: f32 = 0.0;
    let mut e1 = map[1] as f32 / escale;
    let mut e2 = map[i2 + 1] as f32 / escale;

    let mut s0 = true;
    let mut s1 = true;
    let mut s2 = true;
    let mut f0: f32 = 1.001;
    let mut f1: f32 = 1.001;
    let mut f2: f32 = 1.002;

    let e_f32 = energy as f32;

    while e_f32 > e2 && l3 != 0 {
        i0 = i1;
        i1 = i2;
        i2 = i3;
        i3 += l3 as usize;
        l3 = map[i3];
        e0 = e1;
        e1 = e2;
        e2 = map[i2 + 1] as f32 / escale;
        s0 = s1;
        s1 = s2;
        s2 = true;
        f0 = f1;
        f1 = f2;
    }

    if s1 {
        f1 = trara2(map, i1 + 2, nl, nb, fistep) / fscale;
    }
    if s2 {
        f2 = trara2(map, i2 + 2, nl, nb, fistep) / fscale;
    }
    s1 = false;
    s2 = false;

    let mut result = f1 + (f2 - f1) * (e_f32 - e1) / (e2 - e1);

    if f2 <= 0.0 && i1 != 0 {
        if s0 {
            f0 = trara2(map, i0 + 2, nl, nb, fistep) / fscale;
        }
        let alt = f0 + (f1 - f0) * (e_f32 - e0) / (e1 - e0);
        result = result.min(alt);
    }

    result.max(0.0) as f64
}

pub(crate) fn aep8_flux(
    energy_mev: f64,
    l_shell: f64,
    b_over_b0: f64,
    particle: Particle,
    solar: SolarCycle,
) -> f64 {
    let (descr, map) = get_model(particle, solar);
    let log_flux = trara1(descr, map, l_shell, b_over_b0, energy_mev);
    if log_flux > 0.0 {
        10.0_f64.powf(log_flux)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ap8_proton_saa() {
        let flux = aep8_flux(10.0, 1.3, 1.5, Particle::Proton, SolarCycle::Max);
        println!("AP8max proton flux at L=1.3, B/B0=1.5, 10 MeV: {flux:.2e}");
        assert!(flux > 0.0, "should have proton flux in inner belt");
    }

    #[test]
    fn test_ae8_electron_outer() {
        let flux = aep8_flux(1.0, 4.0, 1.0, Particle::Electron, SolarCycle::Max);
        println!("AE8max electron flux at L=4.0, B/B0=1.0, 1 MeV: {flux:.2e}");
        assert!(flux > 0.0, "should have electron flux in outer belt");
    }

    #[test]
    fn test_no_flux_outside_belts() {
        let flux = aep8_flux(10.0, 10.0, 1.0, Particle::Proton, SolarCycle::Max);
        println!("AP8max proton flux at L=10.0: {flux:.2e}");
        assert!(flux == 0.0 || flux < 1.0, "very little proton flux at L=10");
    }
}
