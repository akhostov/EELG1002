SELECT
       object_id
      , ra
      , dec
      , n387_kronflux_mag
      , n387_kronflux_magerr
      , n387_kronflux_radius
      , n816_kronflux_mag
      , n816_kronflux_magerr
      , n921_kronflux_mag
      , n921_kronflux_magerr
      , n1010_kronflux_mag
      , n1010_kronflux_magerr
      , g_kronflux_mag
      , g_kronflux_magerr
      , r_kronflux_mag
      , r_kronflux_magerr
      , i_kronflux_mag
      , i_kronflux_magerr
      , z_kronflux_mag
      , z_kronflux_magerr
      , y_kronflux_mag
      , y_kronflux_magerr
    FROM
       pdr3_dud_rev.forced
       JOIN pdr3_dud_rev.forced2 USING (object_id)
    WHERE
         coneSearch(coord, 150.13463, 2.85315, 1.0)
    LIMIT 1
;



SELECT
       object_id
      , ra
      , dec
      , n387_kronflux_mag
      , n387_kronflux_magsigma
      , n387_kronflux_flag
      , n387_kronflux_flag_used_psf_radius
      , n527_kronflux_mag
      , n527_kronflux_magsigma
      , n718_kronflux_mag
      , n718_kronflux_magsigma       
      , n816_kronflux_mag
      , n816_kronflux_magsigma
      , n921_kronflux_mag
      , n921_kronflux_magsigma
      , i945_kronflux_mag
      , i945_kronflux_magsigma
      , n973_kronflux_mag
      , n973_kronflux_magsigma
      , g_kronflux_mag
      , g_kronflux_magsigma
      , r_kronflux_mag
      , r_kronflux_magsigma
      , i_kronflux_mag
      , i_kronflux_magsigma
      , z_kronflux_mag
      , z_kronflux_magsigma
      , y_kronflux_mag
      , y_kronflux_magsigma
    FROM
       chorus_pdr1.meas
       JOIN chorus_pdr1.meas3 USING (object_id)
    WHERE
         coneSearch(coord, 150.13463, 2.85315, 1.0)
    LIMIT 1
;

