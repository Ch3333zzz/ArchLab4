(defun fib (n)
  (if (<= n 1)
      n
      (+ (call fib (- n 1)) (call fib (- n 2)))))

(print (call fib 6))
