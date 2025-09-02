(defun ie (n)
  (if (<= n 1)
      n
      (+ (call ie (- n 1)) (call ie (- n 2)))))

(print (call ie 6))
