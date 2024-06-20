(deftemplate Data
  (slot prod1AAM)
  (slot prod2AAM)
  (slot react1AAM)
  (slot react2AAM)
  (slot category))


(defrule Regle1
  (Data (react1AAM ?react1AAM) (prod2AAM ?prod2AAM) (prod1AAM ?prod1AAM))
  (test (and (<= ?react1AAM 13.0) (<= ?prod2AAM 1313647493120.0) (> ?prod1AAM 13.0) (<= ?prod2AAM 966171131904.0) (> ?prod2AAM 66114412.0)))
=>
  (assert (class 1)))

(defrule Regle2
  (Data (react1AAM ?react1AAM) (prod1AAM ?prod1AAM) (prod2AAM ?prod2AAM) (react2AAM ?react2AAM))
  (test (and (> ?react1AAM 13.0) (> ?prod1AAM 13.0) (> ?prod2AAM 66117496.0) (<= ?prod2AAM 2666141952507904.0) (> ?react2AAM 356611145728.0)))
=>
  (assert (class 2)))

(defrule Regle3
  (Data (react1AAM ?react1AAM) (prod2AAM ?prod2AAM) (prod1AAM ?prod1AAM) (react2AAM ?react2AAM))
  (test (and (<= ?react1AAM 13.0) (> ?prod2AAM 1313647493120.0) (> ?prod1AAM 13.0) (<= ?prod2AAM 8.161414022567035) (<= ?react2AAM 3.566171153288397)))
=>
  (assert (class 2)))

(defrule Regle4
  (Data (react1AAM ?react1AAM) (prod1AAM ?prod1AAM) (prod2AAM ?prod2AAM) (react2AAM ?react2AAM))
  (test (and (> ?react1AAM 13.0) (> ?prod1AAM 13.0) (> ?prod2AAM 66117496.0) (<= ?prod2AAM 2666141952507904.0) (<= ?react2AAM 356611145728.0)))
=>
  (assert (class 2)))

(defrule Regle5
  (Data (react1AAM ?react1AAM) (prod2AAM ?prod2AAM) (prod1AAM ?prod1AAM))
  (test (and (<= ?react1AAM 13.0) (> ?prod2AAM 1313647493120.0) (<= ?prod1AAM 13.0) (> ?prod2AAM 66661436620800.0) (<= ?prod2AAM 8166171375304704.0)))
=>
  (assert (class 3)))

(defrule Regle6
  (Data (react1AAM ?react1AAM) (prod2AAM ?prod2AAM) (prod1AAM ?prod1AAM) (react2AAM ?react2AAM))
  (test (and (<= ?react1AAM 13.0) (> ?prod2AAM 1313647493120.0) (> ?prod1AAM 13.0) (<= ?prod2AAM 8.161414022567035) (> ?react2AAM 3.566171153288397)))
=>
  (assert (class 1)))

(defrule Regle7
  (Data (react1AAM ?react1AAM) (prod2AAM ?prod2AAM) (prod1AAM ?prod1AAM) (react2AAM ?react2AAM))
  (test (and (<= ?react1AAM 13.0) (<= ?prod2AAM 1313647493120.0) (<= ?prod1AAM 13.0) (<= ?react2AAM 966114410496.0) (> ?react2AAM 53136663040.0)))
=>
  (assert (class 1)))

(defrule Regle8
  (Data (react1AAM ?react1AAM) (prod2AAM ?prod2AAM) (prod1AAM ?prod1AAM) (react2AAM ?react2AAM))
  (test (and (<= ?react1AAM 13.0) (<= ?prod2AAM 1313647493120.0) (<= ?prod1AAM 13.0) (<= ?react2AAM 966114410496.0) (<= ?react2AAM 53136663040.0)))
=>
  (assert (class 2)))

(defrule Regle9
  (Data (react1AAM ?react1AAM) (prod2AAM ?prod2AAM) (prod1AAM ?prod1AAM) (react2AAM ?react2AAM))
  (test (and (<= ?react1AAM 13.0) (> ?prod2AAM 1313647493120.0) (<= ?prod1AAM 13.0) (<= ?prod2AAM 66661436620800.0) (<= ?react2AAM 531366999490560.0)))
=>
  (assert (class 3)))

(defrule Regle10
  (Data (react1AAM ?react1AAM) (prod2AAM ?prod2AAM) (prod1AAM ?prod1AAM) (react2AAM ?react2AAM))
  (test (and (<= ?react1AAM 13.0) (<= ?prod2AAM 1313647493120.0) (<= ?prod1AAM 13.0) (> ?react2AAM 966114410496.0) (<= ?prod2AAM 666942406656.0)))
=>
  (assert (class 1)))

(defrule Regle11
  (Data (react1AAM ?react1AAM) (prod1AAM ?prod1AAM) (react2AAM ?react2AAM) (prod2AAM ?prod2AAM))
  (test (and (> ?react1AAM 13.0) (<= ?prod1AAM 13.0) (> ?react2AAM 966114770944.0) (> ?prod2AAM 1.7661699553230848)))
=>
  (assert (class 4)))

(defrule Regle12
  (Data (react1AAM ?react1AAM) (prod2AAM ?prod2AAM) (prod1AAM ?prod1AAM))
  (test (and (<= ?react1AAM 13.0) (> ?prod2AAM 1313647493120.0) (<= ?prod1AAM 13.0) (> ?prod2AAM 66661436620800.0) (> ?prod2AAM 8166171375304704.0)))
=>
  (assert (class 3)))

(defrule Regle13
  (Data (react1AAM ?react1AAM) (prod1AAM ?prod1AAM) (prod2AAM ?prod2AAM))
  (test (and (> ?react1AAM 13.0) (> ?prod1AAM 13.0) (> ?prod2AAM 66117496.0) (> ?prod2AAM 2666141952507904.0) (> ?prod2AAM 1.7661395147423744)))
=>
  (assert (class 3)))

(defrule Regle14
  (Data (react1AAM ?react1AAM) (prod2AAM ?prod2AAM) (prod1AAM ?prod1AAM) (react2AAM ?react2AAM))
  (test (and (<= ?react1AAM 13.0) (<= ?prod2AAM 1313647493120.0) (<= ?prod1AAM 13.0) (> ?react2AAM 966114410496.0) (> ?prod2AAM 666942406656.0)))
=>
  (assert (class 1)))

(defrule Regle15
  (Data (react1AAM ?react1AAM) (prod2AAM ?prod2AAM) (prod1AAM ?prod1AAM))
  (test (and (<= ?react1AAM 13.0) (<= ?prod2AAM 1313647493120.0) (> ?prod1AAM 13.0) (<= ?prod2AAM 966171131904.0) (<= ?prod2AAM 66114412.0)))
=>
  (assert (class 1)))

(defrule Regle16
  (Data (react1AAM ?react1AAM) (prod2AAM ?prod2AAM) (prod1AAM ?prod1AAM) (react2AAM ?react2AAM))
  (test (and (<= ?react1AAM 13.0) (> ?prod2AAM 1313647493120.0) (<= ?prod1AAM 13.0) (<= ?prod2AAM 66661436620800.0) (> ?react2AAM 531366999490560.0)))
=>
  (assert (class 1)))

(defrule Regle17
  (Data (react1AAM ?react1AAM) (prod1AAM ?prod1AAM) (prod2AAM ?prod2AAM))
  (test (and (> ?react1AAM 13.0) (> ?prod1AAM 13.0) (> ?prod2AAM 66117496.0) (> ?prod2AAM 2666141952507904.0) (<= ?prod2AAM 1.7661395147423744)))
=>
  (assert (class 3)))

(defrule Regle18
  (Data (react1AAM ?react1AAM) (prod2AAM ?prod2AAM) (prod1AAM ?prod1AAM))
  (test (and (<= ?react1AAM 13.0) (> ?prod2AAM 1313647493120.0) (> ?prod1AAM 13.0) (> ?prod2AAM 8.161414022567035)))
=>
  (assert (class 2)))

(defrule Regle19
  (Data (react1AAM ?react1AAM) (prod1AAM ?prod1AAM) (react2AAM ?react2AAM))
  (test (and (> ?react1AAM 13.0) (<= ?prod1AAM 13.0) (<= ?react2AAM 966114770944.0)))
=>
  (assert (class 3)))

(defrule Regle20
  (Data (react1AAM ?react1AAM) (prod1AAM ?prod1AAM) (prod2AAM ?prod2AAM) (react2AAM ?react2AAM))
  (test (and (> ?react1AAM 13.0) (> ?prod1AAM 13.0) (<= ?prod2AAM 66117496.0) (> ?react2AAM 17661441024.0)))
=>
  (assert (class 3)))

(defrule Regle21
  (Data (react1AAM ?react1AAM) (prod1AAM ?prod1AAM) (react2AAM ?react2AAM) (prod2AAM ?prod2AAM))
  (test (and (> ?react1AAM 13.0) (<= ?prod1AAM 13.0) (> ?react2AAM 966114770944.0) (<= ?prod2AAM 1.7661699553230848) (<= ?prod2AAM 187138855403520.0)))
=>
  (assert (class 2)))

(defrule Regle22
  (Data (react1AAM ?react1AAM) (prod1AAM ?prod1AAM) (prod2AAM ?prod2AAM) (react2AAM ?react2AAM))
  (test (and (> ?react1AAM 13.0) (> ?prod1AAM 13.0) (<= ?prod2AAM 66117496.0) (<= ?react2AAM 17661441024.0) (<= ?react2AAM 5713643712.0)))
=>
  (assert (class 3)))

(defrule Regle23
  (Data (react1AAM ?react1AAM) (prod2AAM ?prod2AAM) (prod1AAM ?prod1AAM))
  (test (and (<= ?react1AAM 13.0) (<= ?prod2AAM 1313647493120.0) (> ?prod1AAM 13.0) (> ?prod2AAM 966171131904.0)))
=>
  (assert (class 2)))

(defrule Regle24
  (Data (react1AAM ?react1AAM) (prod1AAM ?prod1AAM) (prod2AAM ?prod2AAM) (react2AAM ?react2AAM))
  (test (and (> ?react1AAM 13.0) (> ?prod1AAM 13.0) (<= ?prod2AAM 66117496.0) (<= ?react2AAM 17661441024.0) (> ?react2AAM 5713643712.0)))
=>
  (assert (class 4)))

(defrule Regle25
  (Data (react1AAM ?react1AAM) (prod1AAM ?prod1AAM) (react2AAM ?react2AAM) (prod2AAM ?prod2AAM))
  (test (and (> ?react1AAM 13.0) (<= ?prod1AAM 13.0) (> ?react2AAM 966114770944.0) (<= ?prod2AAM 1.7661699553230848) (> ?prod2AAM 187138855403520.0)))
=>
  (assert (class 4)))
