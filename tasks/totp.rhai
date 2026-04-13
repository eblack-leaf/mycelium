// @param name: Name of the TOTP secret record to look up

let rows = query(`SELECT secret FROM totp_secret WHERE name = '${params["name"]}'`);
let secret = rows[0].result[0].secret;
let code = shell("oathtool", ["--totp", "-b", secret]);

#{ code: code.trim(), name: params["name"] }
