class prog_upd:
    prog=0.0
    chnl=0
    chnls=1
    i=0
    tr_prog_cb=lambda p,c,cs:print(str(p+c))
    @staticmethod
    def setprog(p):
        prog_upd.prog=p
        prog_upd.i+=1
        prog_upd.tr_prog_cb(prog_upd.prog,prog_upd.chnl,prog_upd.chnls)
        #print(str(prog_upd.prog+prog_upd.chnl))
    @staticmethod
    def setchnl(c,cs):
        prog_upd.chnl=c
        prog_upd.chnls=cs
        #prog_upd.tr_prog_cb(prog_upd.prog,prog_upd.chnl)
