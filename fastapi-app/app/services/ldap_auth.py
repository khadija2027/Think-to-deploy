from typing import Optional, Dict, List
from ldap3 import Server, Connection, ALL, SUBTREE, Tls
import ssl
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

class LDAPAuth:
    def __init__(self):
        """Initialisation de la connexion LDAP avec ou sans TLS/SSL"""
        print(f"🔄 INIT LDAPAuth - Server: {settings.LDAP_SERVER}:{settings.LDAP_PORT}")
        logger.info(f"🔗 Tentative de connexion LDAP{'S' if settings.LDAP_USE_TLS else ''} à {settings.LDAP_SERVER}:{settings.LDAP_PORT}")
        
        if settings.LDAP_USE_TLS:
            # ============================================
            # CONFIGURATION TLS POUR CERTIFICAT AUTO-SIGNÉ
            # ============================================
            tls_configuration = Tls(
                validate=ssl.CERT_NONE,           # Accepter certificat auto-signé
                version=ssl.PROTOCOL_TLSv1_2,     # TLS 1.2 minimum
            )
            
            # ============================================
            # SERVEUR LDAPS (Port 636, SSL activé)
            # ============================================
            self.server = Server(
                settings.LDAP_SERVER,
                port=settings.LDAP_PORT,
                use_ssl=True,                     # SSL/TLS activé
                tls=tls_configuration,
                get_info=ALL
            )
        else:
            # ============================================
            # SERVEUR LDAP (Port 389, sans SSL)
            # ============================================
            self.server = Server(
                settings.LDAP_SERVER,
                port=settings.LDAP_PORT,
                use_ssl=False,                    # Pas de SSL
                get_info=ALL
            )
        
        self.user_base_dn = settings.LDAP_USER_DN
        self.group_base_dn = settings.LDAP_GROUP_DN
        
        logger.info(f"🔗 Tentative de connexion LDAP{'S' if settings.LDAP_USE_TLS else ''} à {settings.LDAP_SERVER}:{settings.LDAP_PORT}")
        
        try:
            # Connexion avec bind automatique
            self.conn = Connection(
                self.server,
                user=settings.LDAP_BIND_DN,
                password=settings.LDAP_BIND_PASSWORD,
                auto_bind=True
            )
            
            logger.info(f"✅ Connexion LDAP{'S' if settings.LDAP_USE_TLS else ''} établie avec succès")
            logger.info(f"   Serveur: {settings.LDAP_SERVER}:{settings.LDAP_PORT}")
            logger.info(f"   TLS: {'Activé avec certificat auto-signé' if settings.LDAP_USE_TLS else 'Désactivé'}")
            logger.info(f"   User Base DN: {self.user_base_dn}")
            logger.info(f"   Group Base DN: {self.group_base_dn}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la connexion LDAP{'S' if settings.LDAP_USE_TLS else ''}: {str(e)}")
            logger.info("💡 Vérifications :")
            logger.info(f"   1. Serveur LDAP accessible : {settings.LDAP_SERVER}")
            logger.info(f"   2. Port LDAP{'S' if settings.LDAP_USE_TLS else ''} ouvert : {settings.LDAP_PORT}")
            if settings.LDAP_USE_TLS:
                logger.info(f"   3. Certificats générés (attendre 30s au premier démarrage)")
            raise

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """
        Authentifie un utilisateur via LDAP et retourne ses informations
        
        Args:
            username: Matricule de l'utilisateur
            password: Mot de passe de l'utilisateur
            
        Returns:
            Dict avec uid, cn, mail, groups si authentification réussie
            None si échec
        """
        try:
            # Construction du DN utilisateur
            user_dn = f"uid={username},{self.user_base_dn}"
            
            logger.info(f"🔍 Tentative d'authentification pour: {username}")
            logger.info(f"   User DN: {user_dn}")
            
            # Créer une nouvelle connexion pour cet utilisateur
            user_conn = Connection(
                self.server,
                user=user_dn,
                password=password
            )
            
            # Tentative de bind avec les credentials de l'utilisateur
            if not user_conn.bind():
                logger.warning(f"❌ Échec d'authentification pour: {username}")
                logger.warning(f"   Raison: {user_conn.result}")
                user_conn.unbind()
                return None
            
            logger.info(f"✅ Authentification réussie pour: {username}")
            
            # Recherche des informations de l'utilisateur
            self.conn.search(
                search_base=self.user_base_dn,
                search_filter=f"(uid={username})",
                search_scope=SUBTREE,
                attributes=["uid", "cn", "mail"]  # Enlever "memberUid" ici
            )
            
            if not self.conn.entries:
                logger.warning(f"⚠️  Utilisateur non trouvé dans l'annuaire: {username}")
                user_conn.unbind()
                return None
            
            entry = self.conn.entries[0]
            
            # Récupérer les groupes avec la méthode spécialisée
            groups = self.get_user_groups(username)
            
            user_info = {
                "uid": str(entry.uid),
                "cn": str(entry.cn),
                "mail": str(entry.mail) if hasattr(entry, 'mail') else None,
                "groups": groups
            }
            
            logger.info(f"📋 Groupes de {username}: {groups}")
            
            # Fermer la connexion utilisateur
            user_conn.unbind()
            
            return user_info
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'authentification: {e}")
            # S'assurer que la connexion est fermée en cas d'erreur
            try:
                if 'user_conn' in locals():
                    user_conn.unbind()
            except:
                pass
            return None

    def list_all_users(self) -> List[Dict]:
        """
        Liste tous les utilisateurs de l'annuaire
        
        Returns:
            Liste de dictionnaires contenant les infos utilisateurs
        """
        try:
            self.conn.search(
                search_base=self.user_base_dn,
                search_filter="(objectClass=inetOrgPerson)",
                search_scope=SUBTREE,
                attributes=["uid", "cn", "mail"]
            )
            
            users = []
            for entry in self.conn.entries:
                users.append({
                    "uid": str(entry.uid),
                    "cn": str(entry.cn),
                    "mail": str(entry.mail) if hasattr(entry, 'mail') else None,
                    "dn": str(entry.entry_dn)
                })
            
            logger.info(f"📋 {len(users)} utilisateurs trouvés dans l'annuaire")
            return users
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des utilisateurs: {e}")
            return []

    def get_user_groups(self, username: str) -> List[str]:
        """
        Récupère les groupes d'un utilisateur pour OpenLDAP posixGroup
        """
        try:
            # Recherche par memberUid (standard OpenLDAP posixGroup)
            self.conn.search(
                search_base=self.group_base_dn,  # Important: chercher dans les groupes!
                search_filter=f"(memberUid={username})",
                search_scope=SUBTREE,
                attributes=["cn"]
            )
            
            groups = [str(entry.cn) for entry in self.conn.entries]
            
            logger.info(f"🔍 Recherche groupes pour {username}: {len(groups)} trouvé(s)")
            
            # Si pas de résultat, vérifier le format exact
            if not groups:
                logger.debug(f"   Aucun groupe trouvé avec memberUid={username}")
                logger.debug(f"   Base DN: {self.group_base_dn}")
                
                # Option: vérifier aussi par DN complet si nécessaire
                # 1. D'abord trouver le DN de l'utilisateur
                self.conn.search(
                    search_base=self.user_base_dn,
                    search_filter=f"(uid={username})",
                    search_scope=SUBTREE,
                    attributes=["dn"]
                )
                
                if self.conn.entries:
                    user_dn = str(self.conn.entries[0].entry_dn)
                    logger.debug(f"   User DN: {user_dn}")
                    
                    # 2. Chercher les groupes avec member (format alternatif)
                    self.conn.search(
                        search_base=self.group_base_dn,
                        search_filter=f"(member={user_dn})",
                        search_scope=SUBTREE,
                        attributes=["cn"]
                    )
                    
                    groups = [str(entry.cn) for entry in self.conn.entries]
                    logger.debug(f"   Groupes par DN: {groups}")
        
            return groups
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des groupes: {e}")
            return []

    def diagnose_groups(self, username: str):
        """
        Méthode de diagnostic pour comprendre la structure des groupes
        """
        try:
            # 1. Vérifier l'utilisateur existe
            self.conn.search(
                search_base=self.user_base_dn,
                search_filter=f"(uid={username})",
                search_scope=SUBTREE,
                attributes=["uid", "dn"]
            )
            
            if not self.conn.entries:
                logger.error(f"❌ Utilisateur {username} non trouvé")
                return
            
            user_entry = self.conn.entries[0]
            user_dn = str(user_entry.entry_dn)
            logger.info(f"✅ Utilisateur trouvé: {user_dn}")
            
            # 2. Lister tous les groupes disponibles
            self.conn.search(
                search_base=self.group_base_dn,
                search_filter="(objectClass=*)",
                search_scope=SUBTREE,
                attributes=["cn", "memberUid", "member", "objectClass"]
            )
            
            logger.info(f"📋 {len(self.conn.entries)} groupes trouvés:")
            for entry in self.conn.entries:
                group_info = f"  - {entry.cn}"
                if hasattr(entry, 'objectClass'):
                    group_info += f" [Classes: {entry.objectClass}]"
                if hasattr(entry, 'memberUid'):
                    members = entry.memberUid
                    if isinstance(members, list):
                        group_info += f" [memberUid: {', '.join(members)}]"
                    else:
                        group_info += f" [memberUid: {members}]"
                if hasattr(entry, 'member'):
                    members = entry.member
                    if isinstance(members, list):
                        group_info += f" [member: {', '.join(members[:2])}{'...' if len(members) > 2 else ''}]"
                    else:
                        group_info += f" [member: {members}]"
                logger.info(group_info)
        
        except Exception as e:
            logger.error(f"❌ Erreur diagnostic: {e}")

    def __del__(self):
        """Fermeture propre de la connexion LDAP"""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.unbind()
                logger.info("🔌 Connexion LDAP fermée")
        except Exception as e:
            logger.error(f"⚠️  Erreur lors de la fermeture de la connexion: {e}")