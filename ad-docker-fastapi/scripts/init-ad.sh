#!/bin/bash

echo "========================================="
echo "Initialisation OpenLDAP"
echo "========================================="

# Attendre que OpenLDAP soit complètement démarré
echo "Attente du démarrage d'OpenLDAP..."
sleep 10

# Fichier LDIF pour la structure de base
cat > /scripts/base-structure.ldif << 'EOF'
# Structure de base pour l'organisation
dn: ou=users,dc=example,dc=com
objectClass: organizationalUnit
ou: users
description: Utilisateurs de l'application

dn: ou=groups,dc=example,dc=com
objectClass: organizationalUnit
ou: groups
description: Groupes de l'application

dn: ou=services,dc=example,dc=com
objectClass: organizationalUnit
ou: services
description: Comptes de service
EOF

# Fichier LDIF pour les utilisateurs
cat > /scripts/users.ldif << 'EOF'
# Utilisateur administrateur de l'application
dn: uid=admin,ou=users,dc=example,dc=com
objectClass: inetOrgPerson
objectClass: posixAccount
objectClass: shadowAccount
uid: admin
cn: Administrateur
sn: System
givenName: Admin
mail: admin@example.com
userPassword: {SSHA}i3Lc8ZjQJcLd1iVrG8Wj7K4mN5s7hJYy
uidNumber: 1000
gidNumber: 1000
homeDirectory: /home/admin
loginShell: /bin/bash

# Utilisateur développeur 1
dn: uid=dev1,ou=users,dc=example,dc=com
objectClass: inetOrgPerson
objectClass: posixAccount
objectClass: shadowAccount
uid: dev1
cn: Développeur Un
sn: Dev
givenName: User1
mail: dev1@example.com
userPassword: {SSHA}i3Lc8ZjQJcLd1iVrG8Wj7K4mN5s7hJYy
uidNumber: 1001
gidNumber: 1001
homeDirectory: /home/dev1
loginShell: /bin/bash

# Utilisateur développeur 2
dn: uid=dev2,ou=users,dc=example,dc=com
objectClass: inetOrgPerson
objectClass: posixAccount
objectClass: shadowAccount
uid: dev2
cn: Développeur Deux
sn: Dev
givenName: User2
mail: dev2@example.com
userPassword: {SSHA}i3Lc8ZjQJcLd1iVrG8Wj7K4mN5s7hJYy
uidNumber: 1002
gidNumber: 1002
homeDirectory: /home/dev2
loginShell: /bin/bash

# Utilisateur test
dn: uid=testuser,ou=users,dc=example,dc=com
objectClass: inetOrgPerson
objectClass: posixAccount
objectClass: shadowAccount
uid: testuser
cn: Utilisateur Test
sn: Test
givenName: User
mail: test@example.com
userPassword: {SSHA}i3Lc8ZjQJcLd1iVrG8Wj7K4mN5s7hJYy
uidNumber: 1003
gidNumber: 1003
homeDirectory: /home/testuser
loginShell: /bin/bash
EOF

# Fichier LDIF pour les groupes
cat > /scripts/groups.ldif << 'EOF'
# Groupe des administrateurs
dn: cn=admins,ou=groups,dc=example,dc=com
objectClass: posixGroup
objectClass: top
cn: admins
gidNumber: 2000
memberUid: admin
description: Administrateurs système

# Groupe des développeurs
dn: cn=developers,ou=groups,dc=example,dc=com
objectClass: posixGroup
objectClass: top
cn: developers
gidNumber: 2001
memberUid: dev1
memberUid: dev2
description: Développeurs d'applications

# Groupe des utilisateurs
dn: cn=users,ou=groups,dc=example,dc=com
objectClass: posixGroup
objectClass: top
cn: users
gidNumber: 2002
memberUid: admin
memberUid: dev1
memberUid: dev2
memberUid: testuser
description: Tous les utilisateurs

# Groupe support
dn: cn=support,ou=groups,dc=example,dc=com
objectClass: posixGroup
objectClass: top
cn: support
gidNumber: 2003
memberUid: dev2
memberUid: testuser
description: Équipe support
EOF

echo "Importation de la structure de base..."
ldapadd -x -H ldap://localhost -D "cn=admin,dc=example,dc=com" -w admin123 -f /scripts/base-structure.ldif

echo "Importation des utilisateurs..."
ldapadd -x -H ldap://localhost -D "cn=admin,dc=example,dc=com" -w admin123 -f /scripts/users.ldif

echo "Importation des groupes..."
ldapadd -x -H ldap://localhost -D "cn=admin,dc=example,dc=com" -w admin123 -f /scripts/groups.ldif

# Tester que tout fonctionne
echo "Vérification de l'importation..."

echo ""
echo "1. Recherche de tous les utilisateurs:"
ldapsearch -x -H ldap://localhost -b "ou=users,dc=example,dc=com" -D "cn=admin,dc=example,dc=com" -w admin123 "(objectClass=*)"

echo ""
echo "2. Recherche de tous les groupes:"
ldapsearch -x -H ldap://localhost -b "ou=groups,dc=example,dc=com" -D "cn=admin,dc=example,dc=com" -w admin123 "(objectClass=*)"

echo ""
echo "========================================="
echo "Initialisation OpenLDAP terminée !"
echo "========================================="
echo ""
echo "Informations de connexion :"
echo "Serveur LDAP: localhost:1389"
echo "Base DN: dc=example,dc=com"
echo "Admin DN: cn=admin,dc=example,dc=com"
echo "Mot de passe admin: admin123"
echo ""
echo "Utilisateurs créés (mot de passe: password123) :"
echo "  - uid=admin,ou=users,dc=example,dc=com"
echo "  - uid=dev1,ou=users,dc=example,dc=com" 
echo "  - uid=dev2,ou=users,dc=example,dc=com"
echo "  - uid=testuser,ou=users,dc=example,dc=com"
echo ""
echo "Groupes créés :"
echo "  - cn=admins,ou=groups,dc=example,dc=com"
echo "  - cn=developers,ou=groups,dc=example,dc=com"
echo "  - cn=users,ou=groups,dc=example,dc=com"
echo "  - cn=support,ou=groups,dc=example,dc=com"
echo ""
echo "Interface web: http://localhost:8080"
echo "Login DN: cn=admin,dc=example,dc=com"
echo "Password: admin123"
echo ""
echo "Pour tester l'authentification :"
echo "ldapwhoami -x -H ldap://localhost:1389 -D \"uid=admin,ou=users,dc=example,dc=com\" -w password123"
echo "========================================="