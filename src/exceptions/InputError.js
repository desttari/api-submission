const ClientError = require("../exceptions/ClientError");

class InputError extends ClientError {
  constructor(message, statusCode = 400) {
    super(message);
    this.statusCode = statusCode;
    this.name = "InputError";
  }
}

module.exports = InputError;
