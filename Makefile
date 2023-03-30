all:
	make -k -C template
	make -k -C US1
	make -k -C US3
	make -k -C V401
	make -k -C V407
	make -k -C V500
	make -k -C V503
	make -k -C V504
	make -k -C V601
	make -k -C V606
	make -k -C V701
	make -k -C V702
	make -k -C V703

template:
	make -C template

US1:
	make -C US1

US3:
	make -C US3

V401:
	make -C V401

V407:
	make -C V407

V500:
	make -C V500

V503:
	make -C V503

V504:
	make -C V504

V601:
	make -C V601

V606:
	make -C V606

V701:
	make -C V701

V702:
	make -C V702

V703:
	make -C V703

clean:
	make -k -C template clean
	make -k -C US1 clean
	make -k -C US3 clean
	make -k -C V401 clean
	make -k -C V407 clean
	make -k -C V500 clean
	make -k -C V503 clean
	make -k -C V504 clean
	make -k -C V601 clean
	make -k -C V606 clean
	make -k -C V701 clean
	make -k -C V702 clean
	make -k -C V703 clean

.PHONY: clean template US1 US3 V401 V407 V500 V503 V504 V601 V606 V701 V702 V703
