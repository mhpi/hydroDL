import torch
import hydroDL.core.logger as logger

log = logger.get_logger("model.wrappers.ModelWrapper")


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        """
        A wrapper that modifies inputs into models to preserve legacy code
        :param model:
        """
        super(ModelWrapper, self).__init__()
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model

    def forward(self, data):
        """
        A forward function that formats data to the legacy models
        Unwraps data from dict --(send tensor in)--> legacy models --(get tensor out)--> wraps output into a dict.
        Collaborating with data_loader
        :param data: A dictionary of data
        :return:
        """
        if hasattr(self.model, "is_legacy"):
            if self.model.is_legacy:
                return self.pre_process(data)
            else:
                """forward function takes in and outputs a dictionary already"""
                return self.model(data)
        else:
            log.debug("Running old model made before latest version")
            return self.pre_process(data)

    def pre_process(self, data):
        """
        A pre-processing function to call the model's forward
        :param data:
        :return:
        """
        if "outModel" in data:
            output = self.model(data["x"], outModel=data["outModel"])
        elif "z" in data:
            output = self.model(data["x"], data["z"])  # what for the tests??
        else:
            output = self.model(data["x"])
        result = {}
        if type(output) is tuple and len(output) == 2:
            """This is for the R2P Parameter Code"""
            result["yP"] = output[0]
            result["Param_R2P"] = output[1]
        elif type(output) is torch.Tensor:
            result["yP"] = output
        elif type(output) is dict:
            result = output
        return result
