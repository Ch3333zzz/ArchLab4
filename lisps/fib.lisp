(defun fib (n)
  (if (<= n 1)
      n
      (+ (call fib (- n 1)) (call fib (- n 2)))))

(setq res (call fib 6))
(print res)
