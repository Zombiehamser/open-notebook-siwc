This is an unofficial, local, text-only prototype based on Open Notebook 1.15.0.
It uses the documented SIWC OAuth flow and public Responses endpoint. It does
not copy OpenAI Devkit code or use browser cookies or another app's login.

## Source and checks

- `siwc_auth.py`: PKCE, identity validation, protected local sessions, rotation,
  account catalog, revocation and explicit proxy enforcement.
- `siwc.py`: Esperanto and LangChain adapters for streamed Responses.
- `../siwc_cli.py`: owner-operated sign-in and model registration.
- `../../tests/test_siwc.py`: synthetic OAuth and inference contract tests.
- `../../Dockerfile.siwc`: stable runtime plus the fork's Python source.

Run `uv run pytest tests/test_siwc.py`, then the normal upstream backend checks.
The regular test workflow includes these tests. The SIWC compatibility workflow
can also merge a selected upstream tag in an ephemeral CI checkout and run checks.
CI never signs into a personal ChatGPT account.

## Local technical setup

Build from the repository root: `docker build -f Dockerfile.siwc -t open-notebook-siwc:prototype .`.
The example `examples/docker-compose-siwc.yml` uses separate app, DB and auth
volumes. Keep its runtime `.env` outside this source tree. Required variables:
`OPEN_NOTEBOOK_ENCRYPTION_KEY`, `SURREAL_PASSWORD`, and
`OPEN_NOTEBOOK_SIWC_PROXY` (for example an owner-controlled HTTP proxy).

Start the example with `docker compose --env-file <private-env> -f <compose> up -d app`.
Use a system browser on the same machine. Start login with
`docker compose --env-file <private-env> -f <compose> --profile login run --rm --service-ports auth`.
Open the printed local `/start` URL and select **Continue with ChatGPT**.
Review the OpenAI consent screen. Passwords and additional authentication stay
on OpenAI's page. The loopback callback is active only during this command.

Sign-in defaults to foreground-only use. Explicitly append `--allow-background`
to the login command's CLI arguments if you consent to transformations and other
background requests. This is consent to the app's jobs, not unlimited usage.
Docker's worker has `OPEN_NOTEBOOK_PROCESS_ROLE=worker`. Custom worker launchers
must set that variable too. Never reuse this single-user session in a shared
Open Notebook deployment serving other people's requests.

After sign-in, use `docker compose ... run --rm auth models` to inspect the live
catalog and `docker compose ... run --rm auth register` to register its language
models. Registration does not change defaults. In Open Notebook select an
available SIWC model for chat, transformations, tools/Ask and large-context text.
Keep embedding and speech models configured separately. For status or logout,
use `docker compose ... run --rm auth status` or `... auth logout`.

The UI connection test lists account models. Individual-model testing performs
a short generation and consumes plan usage. Model slugs come from the account;
availability may differ from the ChatGPT website. Background permission is a
local saved setting; OpenAI's separate plan-use permission and limits still apply.

The current prototype does not support hosted tools, tool calling, audio,
multimodal inputs, or podcast outline/transcript generation. These paths fail
explicitly. Document extraction and retrieval remain Open Notebook's own work.
Selected text/excerpts and questions go to OpenAI for generation. The integration
does not retrieve your ChatGPT conversation history.

Tokens are stored only in the owner-controlled `siwc_auth` volume (directory
0700, files 0600). Keep it out of Git, CI, support transcripts and ordinary
database backups. Registration metadata in the app DB contains no OAuth tokens.
Retain the stable host ID and issued client/account mapping. Logout attempts
remote revocation and clears local tokens; if revocation cannot be confirmed,
disconnect the app in ChatGPT Settings. Each user must connect their own account.

## Protocol and license boundaries

Reference: [official OSS SIWC guide](https://developers.openai.com/siwc/token-sharing-open-source),
[SIWC terms](https://openai.com/policies/sign-in-with-chatgpt-terms/).
Original Open Notebook remains MIT; preserve its LICENSE and copyright notice.
This independent integration remains subject to OpenAI service terms and account
eligibility. No OpenAI endorsement is implied. The separately licensed SIWC
Devkit is not bundled or relicensed. Do not turn this provider into a general
subscription gateway, pool accounts or bypass limits. Requests require an
explicit configured proxy; this is not a universal network firewall for arbitrary
third-party code running in the container.
