
    (setq A_hi 1)
    (setq A_lo 4294967295)
    (setq B_hi 0)
    (setq B_lo 1)

    (setq lo_sum (+ A_lo B_lo))
    (if (>= lo_sum 4294967296)
      (progn
        (setq carry 1)
        (setq lo (- lo_sum 4294967296))
      )
      (progn
        (setq carry 0)
        (setq lo lo_sum)
      )
    )

    (setq hi_sum (+ A_hi B_hi carry))
    (setq hi hi_sum)

    (print hi)
    (print lo)