from importlib.machinery import ExtensionFileLoader
from django.urls import path
from . import views


app_name = 'app'

urlpatterns = [
    path("", views.index_view, name='index'),
    path("aspetos/", views.aspetos_view, name='aspetos'),
    path("cenario/", views.cenario_view, name='cenario'),
    path("diagrama/", views.diagrama_view, name='diagrama'),
    path("planeamento/", views.planeamento_view, name='planeamento'),
    path("plan2/", views.plan2_view, name='plan2'),
    path("plan3/", views.plan3_view, name='plan3'),
    path("trafego/", views.trafego_view, name='trafego'),
    path("erlangB/", views.erlangB_view, name='erlangB'),
    path("canais/", views.canais_view, name='canais'),
    path("calctraf/", views.calctraf_view, name='calctraf'),
    path("topicos/", views.topicos_view, name='topicos'),
    path("Conversor/", views.conversor_view, name='conversor'),
    path("nyquist/", views.nyquist_view, name='nyquist'),
    path("shannon/", views.shannon_view, name='shannon'),
    path("meios/", views.meios_view, name='meios'),
    path("sinais/", views.sinais_view, name='sinais'),
    path("mpsk/", views.mpsk_view, name='mpsk'),
    path("mqam/", views.mqam_view, name='mqam'),
    path("modulacoes/", views.modulacoes_view, name='modulacoes'),
    path("edita_topicos", views.edita_topicos_view, name='edita_topicos'),
    path("novo_topico/", views.novo_topico_view, name='novo_topico'),
    path("edita_topico/<int:topico_id>", views.edita_topico_view, name='edita_topico'),
    path("apaga_topico/<int:topico_id>", views.apaga_topico_view, name='apaga_topico'),
    path("novo_conteudo/", views.novo_conteudo_view, name='novo_conteudo'),
    path("edita_conteudo/<int:conteudo_id>", views.edita_conteudo_view, name='edita_conteudo'),
    path("apaga_conteudo/<int:conteudo_id>", views.apaga_conteudo_view, name='apaga_conteudo'),
]