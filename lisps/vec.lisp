(defun irq-handler ()
  (setq ch (read))
  (out ch)
  (ret)
)

(set-interrupt-vector irq-handler)
(ei)

(while 1
  (progn
    (setq _tmp 0)
  )
)
