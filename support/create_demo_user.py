#!/usr/bin/env python3
"""Create an Autonomous Database user using the configured administrator.

Run from the repository root inside the OML4Py client container:
    python3 support/create_demo_user.py
    python3 support/create_demo_user.py --user EMBEDDING_DEMO --sql 'SELECT USER FROM dual'

This performs real database changes. Existing users are never modified.
"""

import argparse
import getpass
import re
import sys
from pathlib import Path

# Allow this script to run directly from the support directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import oracledb
from support.demo_config import configure_wallet, load_config


def username_value(value):
    value = value.upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,29}", value) or value == "ADMIN":
        raise argparse.ArgumentTypeError("Use a non-ADMIN username with 1-30 letters, numbers, or underscores, beginning with a letter.")
    return value


def quota_value(value):
    value = value.upper()
    if not re.fullmatch(r"[1-9][0-9]*[MG]", value):
        raise argparse.ArgumentTypeError("Quota must be a positive size in M or G, such as 2G.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", type=username_value, default=None, help="New database username (default: database.user in config/config.toml)")
    parser.add_argument("--quota", type=quota_value, default="2G", help="Quota on the DATA tablespace (default: 2G)")
    parser.add_argument("--sql", default=None, help="One SQL statement to run as ADMIN instead of the default table creation; omit a trailing semicolon")
    args = parser.parse_args()
    config = load_config()
    configure_wallet(config, check_files=True)
    try:
        args.user = username_value(args.user or config["database"]["user"])
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(f"Invalid database.user: {exc}")
    admin_user = config["database"]["admin"]
    if args.user.upper() == admin_user.upper():
        raise SystemExit("The demo user and administrator account must be different.")
    dsn = config["database"]["dsn"]
    if dsn == "database-name_low":
        raise SystemExit("Set database.dsn in config/config.toml first.")
    admin_password = getpass.getpass(f"{admin_user} database password: ")
    new_password = getpass.getpass(f"New password for {args.user}: ")
    confirmation = getpass.getpass("Confirm new password: ")
    if not new_password or new_password != confirmation:
        raise SystemExit("The new passwords do not match or are empty.")
    if '"' in new_password or any(ord(char) < 32 for char in new_password):
        raise SystemExit('The new password cannot contain a double quote or control characters.')
    oracledb.init_oracle_client()
    with oracledb.connect(user=admin_user, password=admin_password, dsn=dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT username FROM all_users WHERE username = :name", name=args.user)
            if cursor.fetchone():
                raise SystemExit(f"User {args.user} already exists. No changes were made.")
            print(f"Creating {args.user} with a {args.quota} DATA quota...", flush=True)
            try:
                # Oracle DDL does not accept binds for identifiers or passwords.
                # The identifier and quota are validated above, and quotes are
                # rejected in the password before using a quoted identifier.
                cursor.execute(f'CREATE USER {args.user} IDENTIFIED BY "{new_password}" QUOTA {args.quota} ON DATA')
            except oracledb.Error as exc:
                # Do not echo the DDL or password if the driver includes SQL in an error.
                error = exc.args[0] if exc.args else None
                code = getattr(error, "full_code", None) or getattr(error, "code", None)
                raise SystemExit(f"Could not create the user (database error {code}). Check password policy and privileges.") from None
            finally:
                new_password = confirmation = None
            try:
                cursor.execute(f"GRANT CREATE SESSION, CREATE TABLE, CREATE MINING MODEL TO {args.user}")
            except oracledb.Error:
                print(f"User {args.user} was created, but the privilege grant failed. Database DDL is not rolled back automatically.", file=sys.stderr, flush=True)
                raise
            print("Granted CREATE SESSION, CREATE TABLE, and CREATE MINING MODEL.")
            statement = args.sql or (
                f"CREATE TABLE {args.user}.DEMO_DOCUMENTS ("
                "id NUMBER PRIMARY KEY, "
                "content VARCHAR2(4000), "
                f"embedding VECTOR({config['model']['dimensions']}, FLOAT32))"
            )
            print(f"Executing SQL as {admin_user}: " + ("custom statement" if args.sql else f"create {args.user}.DEMO_DOCUMENTS"), flush=True)
            try:
                cursor.execute(statement)
            except oracledb.Error:
                print(f"User {args.user} and its grants were created, but the SQL statement failed; DDL is not rolled back automatically.", file=sys.stderr, flush=True)
                raise
            if cursor.description:
                print(" | ".join(column[0] for column in cursor.description))
                for row in cursor:
                    print(" | ".join(str(value) for value in row))
            else:
                connection.commit()
                print("Statement completed.")
            print(f"Set database.user = \"{args.user}\" in config/config.toml before importing the model.")


if __name__ == "__main__":
    main()
