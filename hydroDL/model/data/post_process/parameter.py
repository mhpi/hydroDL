"""a file to hold parameter post processing code"""


def post_process(results, inputs, settings):
    if settings["outModel"] is None:
        results["Parameters_R2P"] = (
            results["Param_R2P"].detach().cpu().numpy().swapaxes(0, 1)
        )
    else:
        results["Parameters_R2P"] = results["Param_R2P"].detach().cpu().numpy()
        hymod_forcing = inputs["xTemp"].detach().cpu().numpy().swapaxes(0, 1)

        runFile = os.path.join(savePath, "hymod_run.csv")
        rf = open(runFile, "a+")

        results["q"] = torch.zeros(hymod_forcing.shape[0], hymod_forcing.shape[1])
        results["evap"] = torch.zeros(hymod_forcing.shape[0], hymod_forcing.shape[1])
        for pix in range(hymod_forcing.shape[0]):
            # model_hymod = rnn.hymod(a=Parameters_R2P[pix,0,0], b=Parameters_R2P[pix,0,1],\
            #     cmax=Parameters_R2P[pix,0,2], rq=Parameters_R2P[pix,0,3],\
            #         rs=Parameters_R2P[pix,0,4], s=Parameters_R2P[pix,0,5],\
            #             slow=Parameters_R2P[pix,0,6],\
            #                 fast=[Parameters_R2P[pix,0,7], Parameters_R2P[pix,0,8], Parameters_R2P[pix,0,9]])
            model_hymod = rnn.hymod(
                a=Parameters_R2P[pix, 0],
                b=Parameters_R2P[pix, 1],
                cmax=Parameters_R2P[pix, 2],
                rq=Parameters_R2P[pix, 3],
                rs=Parameters_R2P[pix, 4],
                s=Parameters_R2P[pix, 5],
                slow=Parameters_R2P[pix, 6],
                fast=[
                    Parameters_R2P[pix, 7],
                    Parameters_R2P[pix, 8],
                    Parameters_R2P[pix, 9],
                ],
            )
            for hymod_t in range(hymod_forcing.shape[1]):
                (
                    results["q"][pix, hymod_t],
                    results["evap"][pix, hymod_t],
                ) = model_hymod.advance(
                    hymod_forcing[pix, hymod_t, 0], hymod_forcing[pix, hymod_t, 1],
                )
                nstepsLst = "{:.5f} {:.5f} {:.5f} {:.5f}".format(
                    hymod_forcing[pix, hymod_t, 0],
                    hymod_forcing[pix, hymod_t, 1],
                    results["q"][pix, hymod_t],
                    results["evap"][pix, hymod_t],
                )
                log.info(nstepsLst)
                rf.write(nstepsLst + "\n")
    return results
