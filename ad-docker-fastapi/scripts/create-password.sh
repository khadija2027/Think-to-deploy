#!/bin/bash
# Générer un mot de passe hashé pour LDAP
# Usage: ./create-password.sh "monmotdepasse"

PASSWORD=$1
if [ -z "$PASSWORD" ]; then
    PASSWORD="password123"
fi

echo "Mot de passe: $PASSWORD"
echo "Hash SSHA: $(slappasswd -s "$PASSWORD")"
echo "Hash MD5: $(slappasswd -h {MD5} -s "$PASSWORD")"
echo "Hash SHA: $(slappasswd -h {SHA} -s "$PASSWORD")"