 (setq n 100)
    (setq sum (/ (* n (+ n 1)) 2))
    (setq sumsq (/ (* n (+ n 1) (+ (* 2 n) 1)) 6))
    (print (- (* sum sum) sumsq))