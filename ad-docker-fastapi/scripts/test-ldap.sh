#!/bin/bash

echo "=== Test de connexion OpenLDAP ==="

# Test 1: Connexion admin
echo "1. Test connexion admin..."
ldapwhoami -x -H ldap://localhost -D "cn=admin,dc=example,dc=com" -w admin123
if [ $? -eq 0 ]; then
    echo "✅ Connexion admin réussie"
else
    echo "❌ Échec connexion admin"
fi

# Test 2: Connexion utilisateur
echo ""
echo "2. Test connexion utilisateur (admin)..."
ldapwhoami -x -H ldap://localhost -D "uid=admin,ou=users,dc=example,dc=com" -w password123
if [ $? -eq 0 ]; then
    echo "✅ Connexion utilisateur réussie"
else
    echo "❌ Échec connexion utilisateur"
fi

# Test 3: Recherche utilisateurs
echo ""
echo "3. Recherche de tous les utilisateurs..."
ldapsearch -x -H ldap://localhost -b "ou=users,dc=example,dc=com" \
  -D "cn=admin,dc=example,dc=com" -w admin123 \
  "(objectClass=*)" uid cn mail 2>/dev/null | grep -E "(^uid:|^cn:|^mail:)"

# Test 4: Recherche groupes avec membres
echo ""
echo "4. Recherche des groupes avec leurs membres..."
ldapsearch -x -H ldap://localhost -b "ou=groups,dc=example,dc=com" \
  -D "cn=admin,dc=example,dc=com" -w admin123 \
  "(objectClass=*)" cn memberUid 2>/dev/null | grep -E "(^cn:|^memberUid:)"

# Test 5: Vérifier l'appartenance aux groupes
echo ""
echo "5. Groupes de l'utilisateur dev1..."
ldapsearch -x -H ldap://localhost -b "ou=groups,dc=example,dc=com" \
  -D "cn=admin,dc=example,dc=com" -w admin123 \
  "(memberUid=dev1)" cn 2>/dev/null | grep "^cn:"

echo ""
echo "=== Tests terminés ==="